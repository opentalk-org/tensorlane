use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use tokio::{
    fs,
    sync::{mpsc, oneshot},
};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, info_span};
use uuid::Uuid;

use super::DataConfig;
use crate::{
    db::fetch_samples,
    loader::Loader,
    prefetch::{LoadedBatch, Prefetcher},
    sampling::{QuerySampler, Sampler},
};

pub struct RunState {
    pub id: Uuid,
    pub(super) cancel_token: CancellationToken,
    validation_batches: Prefetcher,
    training_batches: Prefetcher,
    run_cache_dir: PathBuf,
}

pub(super) struct BatchRequest {
    pub(super) validation: bool,
    pub(super) reply: oneshot::Sender<Result<Option<LoadedBatch>>>,
}

impl RunState {
    pub async fn new(
        id: Uuid,
        database: &clickhouse::Client,
        loader: Arc<dyn Loader>,
        cache_dir: &Path,
        config: &DataConfig,
    ) -> Result<Self> {
        info!(run = %id, dataset = %config.dataset_id, "initializing run");

        let validation_sampler = load_sampler(
            database,
            config,
            "validation",
            config.validation.max_seconds,
            vec![(
                "sample_size",
                json!(u64::try_from(config.validation.samples)?),
            )],
        )
        .await?;
        let training_sampler = load_sampler(
            database,
            config,
            "training",
            config.training_max_seconds(),
            vec![
                ("validation_ids", json!(validation_sampler.audio_ids())),
                (
                    "stage_batches",
                    json!(
                        config
                            .training
                            .iter()
                            .map(|s| s.batches)
                            .collect::<Vec<_>>()
                    ),
                ),
                (
                    "stage_seconds",
                    json!(
                        config
                            .training
                            .iter()
                            .map(|s| s.max_seconds as f64)
                            .collect::<Vec<_>>()
                    ),
                ),
            ],
        )
        .await?;
        let expected_batches: u64 = config.training.iter().map(|stage| stage.batches).sum();
        ensure!(
            training_sampler.len() as u64 == expected_batches,
            "query returned {} batches, expected {expected_batches}",
            training_sampler.len()
        );

        let cache_dir = cache_dir.join(id.to_string());
        let streams: [(&str, Box<dyn Sampler>); 2] = [
            ("training", Box::new(training_sampler)),
            ("validation", Box::new(validation_sampler.repeat())),
        ];
        // Prepare both directories before starting tasks so setup errors leave no workers running.
        for split in streams.each_ref().map(|(split, _)| *split) {
            fs::create_dir_all(cache_dir.join("data").join(split)).await?;
        }
        let cancel_token = CancellationToken::new();
        let [training_batches, validation_batches] = streams.map(|(split, sampler)| {
            Prefetcher::spawn(
                sampler,
                loader.clone(),
                cache_dir.join("data").join(split),
                cancel_token.clone(),
                info_span!("prefetcher", run = %id, split),
            )
        });

        Ok(RunState {
            id,
            cancel_token,
            training_batches,
            validation_batches,
            run_cache_dir: cache_dir,
        })
    }

    pub async fn next_batch(&mut self, validation: bool) -> Result<Option<LoadedBatch>> {
        let batches = if validation {
            &mut self.validation_batches
        } else {
            &mut self.training_batches
        };
        batches.next_batch().await
    }

    pub(super) async fn handle_requests(mut self, mut rx: mpsc::Receiver<BatchRequest>) {
        loop {
            let BatchRequest { validation, reply } = tokio::select! {
                biased;
                () = self.cancel_token.cancelled() => break,
                req = rx.recv() => match req {
                    Some(req) => req,
                    None => break,
                },
            };
            let cancel_token = self.cancel_token.clone();
            let batch = tokio::select! {
                biased;
                () = cancel_token.cancelled() => break,
                batch = self.next_batch(validation) => batch,
            };
            let _ = reply.send(batch);
        }

        self.finish().await;
    }

    pub(super) async fn finish(self) {
        self.cancel_token.cancel();
        tokio::join!(
            self.validation_batches.finish(),
            self.training_batches.finish(),
        );
        if let Err(err) = fs::remove_dir_all(&self.run_cache_dir).await
            && err.kind() != std::io::ErrorKind::NotFound
        {
            error!(run = %self.id, error = %err, path = %self.run_cache_dir.display(), "removing run cache failed");
        }
        debug!("run finished");
    }
}

async fn load_sampler(
    database: &clickhouse::Client,
    config: &DataConfig,
    split: &str,
    max_duration: f32,
    mut params: Vec<(&str, Value)>,
) -> Result<QuerySampler> {
    params.extend([
        ("dataset_id", json!(config.dataset_id)),
        ("seed", json!(config.seed)),
        ("max_duration", json!(max_duration as f64)),
        ("max_text", json!(u64::try_from(config.max_text_tokens)?)),
    ]);
    let sql = config
        .queries
        .get(split)
        .with_context(|| format!("missing data_config.queries.{split}"))?;
    let rows = fetch_samples(database, sql, &params).await?;
    info!(split, rows = rows.len(), "fetched query rows");
    QuerySampler::new(rows, &config.plbert_languages)
}

#[cfg(test)]
mod tests {
    use super::load_sampler;
    use crate::run::DataConfig;

    #[tokio::test]
    async fn missing_queries_return_configuration_errors_without_connecting() {
        let mut document: serde_json::Value =
            serde_json::from_str(include_str!("../../../sample-configs.json")).unwrap();
        document["data_config"]["queries"] = serde_json::json!({});
        let config: DataConfig = serde_json::from_value(document["data_config"].clone()).unwrap();
        for split in ["training", "validation"] {
            let result = load_sampler(
                &clickhouse::Client::default(),
                &config,
                split,
                150.0,
                vec![],
            )
            .await;
            let error = result.err().expect("missing query must fail");
            assert_eq!(
                error.to_string(),
                format!("missing data_config.queries.{split}")
            );
        }
    }
}
