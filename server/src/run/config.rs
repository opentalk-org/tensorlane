use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Deserialize, Serialize)]
pub struct DataConfig {
    pub queries: HashMap<String, String>,
    pub dataset_id: Uuid,
    pub asset_type: String,
    pub seed: u64,
    pub max_text_tokens: i32,
    #[serde(default)]
    pub plbert_languages: Vec<String>,

    #[serde(default)]
    pub assets: std::collections::HashMap<String, AssetConfig>,
    pub validation: ValidationConfig,
    pub training: Vec<SequenceConfig>,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct AssetConfig {
    pub object: String,
    pub entrypoint: Option<String>,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct ValidationConfig {
    pub samples: i64,
    pub max_seconds: f32,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct SequenceConfig {
    pub batches: u64,
    pub max_seconds: f32,
}

impl DataConfig {
    pub fn training_max_seconds(&self) -> f32 {
        self.training
            .iter()
            .map(|s| s.max_seconds)
            .fold(0.0, f32::max)
    }
}
