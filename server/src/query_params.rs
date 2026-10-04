use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;
use std::collections::BTreeMap;

pub fn bind(
    mut query: clickhouse::query::Query,
    sql: &str,
    params: &BTreeMap<String, Value>,
) -> Result<clickhouse::query::Query> {
    let types = parameter_types(sql)?;
    for name in types.keys() {
        ensure!(params.contains_key(name), "missing parameter: {name}");
    }
    for (name, value) in params {
        ensure!(
            !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'),
            "invalid query parameter name: {name}"
        );
        let kind = types.get(name).map(String::as_str).unwrap_or("");
        let text = render(value, kind, false)
            .with_context(|| format!("encoding query parameter {name}"))?;
        query = query.with_setting(format!("param_{name}"), text);
    }
    Ok(query)
}

fn escape(value: &str) -> String {
    let mut output = String::new();
    for c in value.chars() {
        match c {
            '\\' => output.push_str("\\\\"),
            '\'' => output.push_str("\\'"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            '\0' => output.push_str("\\0"),
            c => output.push(c),
        }
    }
    output
}

fn arguments(kind: &str) -> Vec<&str> {
    let Some(start) = kind.find('(') else {
        return Vec::new();
    };
    let Some(end) = kind.rfind(')') else {
        return Vec::new();
    };
    let text = &kind[start + 1..end];
    let mut parts = Vec::new();
    let (mut depth, mut quoted, mut escaped, mut start) = (0i32, false, false, 0);
    for (i, c) in text.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if c == '\\' && quoted {
            escaped = true;
            continue;
        }
        if c == '\'' {
            quoted = !quoted;
        }
        if quoted {
            continue;
        }
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(text[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    if !text.is_empty() {
        parts.push(text[start..].trim());
    }
    parts
}

fn render(value: &Value, kind: &str, nested: bool) -> Result<String> {
    if let Some(object) = value.as_object()
        && let Some(raw) = object.get("$clickhouse")
    {
        ensure!(object.len() == 1, "$clickhouse must be the only field");
        return Ok(raw
            .as_str()
            .context("$clickhouse must contain native ClickHouse text")?
            .to_owned());
    }
    let base = kind.split('(').next().unwrap_or("").trim();
    let args = arguments(kind);
    if matches!(base, "Nullable" | "LowCardinality") && !value.is_null() {
        return render(value, args.first().copied().unwrap_or(""), nested);
    }
    Ok(match value {
        Value::Null => if nested { "NULL" } else { "\\N" }.into(),
        Value::Bool(v) => if *v { "1" } else { "0" }.into(),
        Value::Number(v) => v.to_string(),
        Value::String(v) => {
            let numeric = base.starts_with("Int")
                || base.starts_with("UInt")
                || base.starts_with("Float")
                || base.starts_with("Decimal")
                || base == "Bool";
            if !nested {
                escape(v)
            } else if numeric {
                v.clone()
            } else {
                format!("'{}'", escape(v))
            }
        }
        Value::Array(items) => {
            let tuple = base == "Tuple";
            if tuple {
                ensure!(
                    items.len() == args.len(),
                    "tuple parameter has wrong field count"
                );
            }
            let mut values = Vec::new();
            for (i, value) in items.iter().enumerate() {
                let kind = if tuple {
                    args[i]
                } else {
                    args.first().copied().unwrap_or("")
                };
                values.push(render(value, kind, true)?);
            }
            if tuple {
                format!("({})", values.join(","))
            } else {
                format!("[{}]", values.join(","))
            }
        }
        Value::Object(items) => {
            if base != "Map" || args.len() != 2 {
                bail!("object parameters require Map or {{\"$clickhouse\":\"native text\"}}");
            }
            let mut values = Vec::new();
            for (key, value) in items {
                values.push(format!(
                    "{}:{}",
                    render(&Value::String(key.clone()), args[0], true)?,
                    render(value, args[1], true)?
                ));
            }
            format!("{{{}}}", values.join(","))
        }
    })
}

fn parameter_types(sql: &str) -> Result<BTreeMap<String, String>> {
    let bytes = sql.as_bytes();
    let mut types = BTreeMap::new();
    let mut i = 0;
    while i < bytes.len() {
        if matches!(bytes[i], b'\'' | b'"' | b'`') {
            let quote = bytes[i];
            i += 1;
            while i < bytes.len() {
                if bytes[i] == b'\\' {
                    i += 2;
                    continue;
                }
                if bytes[i] == quote {
                    if bytes.get(i + 1) == Some(&quote) {
                        i += 2;
                        continue;
                    }
                    i += 1;
                    break;
                }
                i += 1;
            }
            continue;
        }
        if bytes[i..].starts_with(b"--") || bytes[i] == b'#' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if bytes[i..].starts_with(b"/*") {
            i += 2;
            while i + 1 < bytes.len() && &bytes[i..i + 2] != b"*/" {
                i += 1;
            }
            i = (i + 2).min(bytes.len());
            continue;
        }
        if bytes[i] == b'{' {
            let start = i + 1;
            let mut end = start;
            let mut quote = false;
            while end < bytes.len() {
                if bytes[end] == b'\\' && quote {
                    end += 2;
                    continue;
                }
                if bytes[end] == b'\'' {
                    quote = !quote;
                }
                if bytes[end] == b'}' && !quote {
                    break;
                }
                end += 1;
            }
            if end < bytes.len() {
                if let Some((name, kind)) = sql[start..end].split_once(':') {
                    let name = name.trim();
                    let kind = kind.trim();
                    if name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
                        && let Some(previous) = types.insert(name.to_owned(), kind.to_owned())
                    {
                        ensure!(
                            previous == kind,
                            "query parameter {name} has inconsistent types"
                        );
                    }
                }
                i = end + 1;
                continue;
            }
        }
        i += 1;
    }
    Ok(types)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn native_parameters_cover_scalars_nested_types_and_precision() {
        assert_eq!(
            render(&json!(-9223372036854775808i64), "Int64", false).unwrap(),
            "-9223372036854775808"
        );
        assert_eq!(
            render(&json!(18446744073709551615u64), "UInt64", false).unwrap(),
            "18446744073709551615"
        );
        assert_eq!(
            render(&json!(null), "Nullable(Int64)", false).unwrap(),
            "\\N"
        );
        assert_eq!(
            render(&json!([null, -3]), "Array(Nullable(Int64))", false).unwrap(),
            "[NULL,-3]"
        );
        assert_eq!(
            render(
                &json!(["a'b", [1, 2]]),
                "Tuple(String, Array(Int64))",
                false
            )
            .unwrap(),
            "('a\\'b',[1,2])"
        );
        assert_eq!(
            render(
                &json!({"x": [1, null]}),
                "Map(String, Array(Nullable(Int64)))",
                false
            )
            .unwrap(),
            "{'x':[1,NULL]}"
        );
        assert_eq!(
            render(
                &json!({"$clickhouse":"123456789012345678901234567890.000000001"}),
                "Decimal256(9)",
                false
            )
            .unwrap(),
            "123456789012345678901234567890.000000001"
        );
        assert_eq!(
            render(&json!("a\n\\b"), "String", false).unwrap(),
            "a\\n\\\\b"
        );
    }

    #[test]
    fn finds_nested_parameter_types_without_reading_comments_or_literals() {
        let types = parameter_types("SELECT '{fake:Int8}', {x:Tuple(String, Array(Int64))}, {x:Tuple(String, Array(Int64))} /* {no:Int8} */ -- {other:Int8}\n").unwrap();
        assert_eq!(types.len(), 1);
        assert_eq!(types["x"], "Tuple(String, Array(Int64))");
        assert!(parameter_types("SELECT {x:Int64}, {x:String}").is_err());
    }
}
