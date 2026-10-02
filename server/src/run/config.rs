use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashMap};
use uuid::Uuid;

#[derive(Clone)]
pub struct Config {
    pub queries: BTreeMap<String, QueryConfig>,
    pub assets: HashMap<String, AssetConfig>,
    pub asset_type: Option<String>,
}
#[derive(Clone)]
pub struct QueryConfig {
    pub sql: String,
    pub params: BTreeMap<String, Value>,
    pub repeat: bool,
    pub batches: Option<u64>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssetConfig {
    pub object: Option<String>,
    pub asset_id: Option<Uuid>,
    pub entrypoint: Option<String>,
}
impl Config {
    pub fn parse(document: &Map<String, Value>) -> Result<Self> {
        let queries = document.get("queries").and_then(Value::as_object)
            .context("config.queries must be a nonempty object (historical configurations cannot be initialized)")?;
        ensure!(!queries.is_empty(), "config.queries must not be empty");
        for name in ["ranks", "num_workers", "prefetch_factor"] {
            if let Some(value) = document.get(name) {
                ensure!(
                    value.as_u64().is_some_and(|count| count > 0),
                    "config.{name} must be a positive integer"
                );
            }
        }
        let mut shared = BTreeMap::new();
        for name in ["dataset_id", "seed"] {
            if let Some(value) = document.get(name) {
                shared.insert(name.to_owned(), value.clone());
            }
        }
        if let Some(params) = document.get("start_params") {
            shared.extend(
                params
                    .as_object()
                    .context("config.start_params must be an object")?
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone())),
            );
        }
        shared.remove("repeat");
        let mut compiled = BTreeMap::new();
        for (name, sql) in queries {
            ensure!(!name.is_empty(), "query name must not be empty");
            ensure!(
                ![
                    "queries",
                    "assets",
                    "start_params",
                    "dataset_id",
                    "seed",
                    "asset_type",
                    "ranks",
                    "num_workers",
                    "prefetch_factor"
                ]
                .contains(&name.as_str()),
                "reserved query name: {name}"
            );
            let sql = sql
                .as_str()
                .filter(|sql| !sql.trim().is_empty())
                .context("query SQL must be a nonempty string")?;
            let mut params = shared.clone();
            let mut repeat = name == "validation";
            let mut batches = None;
            if let Some(section) = document.get(name) {
                let section = section.as_object().with_context(|| format!("config.{name} must be a single parameter object; stage arrays are unsupported"))?;
                for (key, value) in section {
                    if key == "repeat" {
                        repeat = value.as_bool().context("repeat must be boolean")?;
                    } else {
                        params.insert(key.clone(), value.clone());
                    }
                }
            }
            if let Some(value) = params.get("batches") {
                batches = Some(
                    value
                        .as_u64()
                        .context("batches must be a nonnegative integer")?,
                );
            }
            compiled.insert(
                name.clone(),
                QueryConfig {
                    sql: sql.to_owned(),
                    params,
                    repeat,
                    batches,
                },
            );
        }
        let assets: HashMap<String, AssetConfig> = document
            .get("assets")
            .map(|value| serde_json::from_value(value.clone()))
            .transpose()?
            .unwrap_or_default();
        for (name, asset) in &assets {
            ensure!(!name.is_empty(), "asset name must not be empty");
            ensure!(
                asset.object.is_some() != asset.asset_id.is_some(),
                "asset {name} requires exactly one of object or asset_id"
            );
            ensure!(
                asset.object.as_ref().is_none_or(|key| !key.is_empty()),
                "asset object must not be empty"
            );
            ensure!(
                asset.asset_id.is_none_or(|id| !id.is_nil()),
                "input asset ID must not be nil"
            );
        }
        let asset_type = document
            .get("asset_type")
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .context("asset_type must be a string")
            })
            .transpose()?;
        Ok(Self {
            queries: compiled,
            assets,
            asset_type,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn binds_precedence_and_keeps_application_fields() {
        let value = json!({"queries":{"training":"SELECT 1", "evaluation":"SELECT 2", "validation":"SELECT 3"},
            "seed":1,"start_params":{"seed":2,"dataset_offset":9},"training":{"seed":3,"batches":4,"repeat":true},"optimizer":{"lr":0.1}});
        let cfg = Config::parse(value.as_object().unwrap()).unwrap();
        assert_eq!(cfg.queries["training"].params["seed"], 3);
        assert_eq!(cfg.queries["training"].params["dataset_offset"], 9);
        assert!(!cfg.queries["training"].params.contains_key("repeat"));
        assert_eq!(cfg.queries["training"].batches, Some(4));
        assert!(cfg.queries["training"].repeat && cfg.queries["validation"].repeat);
        assert!(!cfg.queries["evaluation"].repeat);
        assert_eq!(value["optimizer"]["lr"], 0.1);
    }
    #[test]
    fn rejects_stage_arrays_reserved_names_and_invalid_assets() {
        for value in [
            json!({"queries":{"training":"SELECT 1"},"training":[{"batches":1}]}),
            json!({"queries":{"assets":"SELECT 1"}}),
            json!({"queries":{}}),
            json!({"queries":{"x":"SELECT 1"},"assets":{"model":{"object":"x","asset_id":Uuid::new_v4()}}}),
        ] {
            assert!(Config::parse(value.as_object().unwrap()).is_err());
        }
    }
    #[test]
    fn validates_runtime_counts_and_reserves_their_names() {
        for name in ["ranks", "num_workers", "prefetch_factor"] {
            let mut value = json!({"queries":{"training":"SELECT 1"}});
            value[name] = json!(2);
            assert!(Config::parse(value.as_object().unwrap()).is_ok());
            for invalid in [json!(0), json!(-1), json!(true), json!(1.5), Value::Null] {
                value[name] = invalid;
                assert!(Config::parse(value.as_object().unwrap()).is_err());
            }
            let mut reserved = json!({"queries":{}});
            reserved["queries"][name] = json!("SELECT 1");
            assert!(Config::parse(reserved.as_object().unwrap()).is_err());
        }
    }
}
