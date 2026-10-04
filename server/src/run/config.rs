use anyhow::{Result, anyhow, ensure};
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
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssetConfig {
    pub object: Option<String>,
    pub asset_id: Option<Uuid>,
    pub entrypoint: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    #[serde(default)]
    tensorlane: TensorlaneConfig,
    queries: Vec<KeyedQuery>,
    #[serde(default, rename = "app")]
    _app: Map<String, Value>,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorlaneConfig {
    ranks: Option<u64>,
    num_workers: Option<u64>,
    prefetch_factor: Option<u64>,
    #[serde(default)]
    assets: HashMap<String, AssetConfig>,
    asset_type: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct KeyedQuery {
    key: String,
    sql: String,
    #[serde(default)]
    params: BTreeMap<String, Value>,
    #[serde(default)]
    repeat: bool,
}

impl Config {
    pub fn parse(document: &Map<String, Value>) -> Result<Self> {
        let document: Document = serde_json::from_value(Value::Object(document.clone()))
            .map_err(|error| anyhow!("invalid run config: {error}"))?;
        ensure!(
            !document.queries.is_empty(),
            "config.queries must not be empty"
        );
        let settings = document.tensorlane;
        for (name, count) in [
            ("ranks", settings.ranks),
            ("num_workers", settings.num_workers),
            ("prefetch_factor", settings.prefetch_factor),
        ] {
            ensure!(
                count.is_none_or(|count| count > 0),
                "config.tensorlane.{name} must be a positive integer"
            );
        }
        let mut compiled = BTreeMap::new();
        for query in document.queries {
            ensure!(
                !query.key.trim().is_empty(),
                "query key must not be empty"
            );
            ensure!(
                !compiled.contains_key(&query.key),
                "duplicate query key: {}",
                query.key
            );
            ensure!(
                !query.sql.trim().is_empty(),
                "query {} SQL must not be empty",
                query.key
            );
            compiled.insert(
                query.key,
                QueryConfig {
                    sql: query.sql,
                    params: query.params,
                    repeat: query.repeat,
                },
            );
        }
        for (name, asset) in &settings.assets {
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
        Ok(Self {
            queries: compiled,
            assets: settings.assets,
            asset_type: settings.asset_type,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(value: Value) -> Result<Config> {
        Config::parse(value.as_object().unwrap())
    }

    #[test]
    fn query_parameters_and_application_settings_are_independent() {
        let cfg = parse(json!({
            "tensorlane": {"num_workers": 2},
            "queries": [
                {"key": "training", "sql": "SELECT 1", "params": {"seed": 3, "batches": 4, "repeat": "SQL input"}},
                {"key": "validation", "sql": "SELECT 2"},
                {"key": "app", "sql": "SELECT 3", "params": {"seed": 9}, "repeat": true}
            ],
            "app": {"num_workers": "opaque", "queries": [1, 2], "seed": 100}
        })).unwrap();
        assert_eq!(cfg.queries["training"].params["seed"], 3);
        assert_eq!(cfg.queries["training"].params["repeat"], "SQL input");
        assert!(!cfg.queries["training"].repeat);
        assert!(!cfg.queries["validation"].repeat);
        assert!(cfg.queries["validation"].params.is_empty());
        assert_eq!(cfg.queries["app"].params["seed"], 9);
        assert!(cfg.queries["app"].repeat);
    }

    #[test]
    fn rejects_ambiguous_or_malformed_queries() {
        for value in [
            json!({"queries": {"training": "SELECT 1"}}),
            json!({"queries": []}),
            json!({"queries": [{"sql": "SELECT 1"}]}),
            json!({"queries": [{"key": " ", "sql": "SELECT 1"}]}),
            json!({"queries": [{"key": "x", "sql": " "}]}),
            json!({"queries": [{"key": "x", "sql": "SELECT 1"}, {"key": "x", "sql": "SELECT 2"}]}),
            json!({"queries": [{"key": "x", "sql": "SELECT 1", "params": []}]}),
            json!({"queries": [{"key": "x", "sql": "SELECT 1", "repeat": "true"}]}),
            json!({"queries": [{"key": "x", "sql": "SELECT 1", "expected_batches": 1}]}),
            json!({"queries": [{"key": "x", "sql": "SELECT 1", "batch_size": 32}]}),
        ] {
            assert!(parse(value.clone()).is_err(), "accepted {value}");
        }
    }

    #[test]
    fn validates_namespaces_and_runtime_counts() {
        for name in ["ranks", "num_workers", "prefetch_factor"] {
            let mut value =
                json!({"tensorlane": {}, "queries": [{"key": "x", "sql": "SELECT 1"}]});
            value["tensorlane"][name] = json!(2);
            assert!(parse(value.clone()).is_ok());
            for invalid in [json!(0), json!(-1), json!(true), json!(1.5)] {
                value["tensorlane"][name] = invalid;
                assert!(parse(value.clone()).is_err());
            }
        }
        for (name, value) in [
            ("training", json!({})),
            ("params", json!({})),
            ("num_workers", json!(2)),
            ("tensorlane", json!({"typo": 1})),
            ("tensorlane", json!([])),
            ("app", json!([])),
        ] {
            let mut document = json!({"queries": [{"key": "x", "sql": "SELECT 1"}]});
            document[name] = value;
            assert!(parse(document).is_err());
        }
    }

    #[test]
    fn assets_belong_to_tensorlane() {
        let id = Uuid::new_v4();
        let mut value = json!({"tensorlane": {"asset_type": "model", "assets": {"model": {"asset_id": id}}},
            "queries": [{"key": "x", "sql": "SELECT 1"}]});
        let cfg = parse(value.clone()).unwrap();
        assert_eq!(cfg.assets["model"].asset_id, Some(id));
        assert_eq!(cfg.asset_type.as_deref(), Some("model"));
        for invalid in [
            json!({}),
            json!({"object": ""}),
            json!({"asset_id": Uuid::nil()}),
            json!({"object": "x", "asset_id": id}),
        ] {
            value["tensorlane"]["assets"]["model"] = invalid;
            assert!(parse(value.clone()).is_err());
        }
    }

    #[test]
    fn bundled_configurations_use_the_keyed_query_schema() {
        for source in [
            include_str!("../../../sample-configs.json"),
            include_str!("../../../sample-configs-stage2.json"),
            include_str!("../../../queries/examples/sample-configs.json"),
            include_str!("../../../queries/examples/sample-configs-stage2.json"),
        ] {
            let value: Value = serde_json::from_str(source).unwrap();
            let cfg = Config::parse(value["config"].as_object().unwrap()).unwrap();
            assert_eq!(cfg.queries.len(), 2);
            assert!(!cfg.queries["training"].repeat);
            assert!(cfg.queries["validation"].repeat);
        }
    }
}
