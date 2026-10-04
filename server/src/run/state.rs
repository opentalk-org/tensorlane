use super::Config;
use crate::{
    db::stream_samples,
    loader::Loader,
    prefetch::{LoadedBatch, Prefetcher},
    sampling::QuerySampler,
};
use anyhow::Result;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::{
    fs,
    sync::{mpsc, oneshot},
};
use tokio_util::sync::CancellationToken;
use tracing::{error, info_span};
use uuid::Uuid;

pub(super) struct BatchRequest {
    pub reply: oneshot::Sender<Result<Option<LoadedBatch>>>,
}
pub struct RunState {
    pub(super) cancel: CancellationToken,
    streams: HashMap<String, Prefetcher>,
    cache: PathBuf,
}
impl RunState {
    pub async fn new(
        id: Uuid,
        database: &clickhouse::Client,
        loader: Arc<dyn Loader>,
        cache: &Path,
        config: &Config,
    ) -> Result<Self> {
        let cache = cache.join(id.to_string());
        fs::create_dir_all(cache.join("plans")).await?;
        let mut plans = Vec::new();
        for (index, (name, query)) in config.queries.iter().enumerate() {
            let rows = stream_samples(database, &query.sql, &query.params);
            let sampler = QuerySampler::create(
                name,
                rows,
                &cache.join("plans").join(format!("{index}.plan")),
                query.repeat,
            )
            .await?;
            plans.push((name.clone(), sampler));
            fs::create_dir_all(cache.join("data").join(index.to_string())).await?;
        }
        let cancel = CancellationToken::new();
        let streams = plans
            .into_iter()
            .enumerate()
            .map(|(index, (name, sampler))| {
                let prefetch = Prefetcher::spawn(
                    Box::new(sampler),
                    loader.clone(),
                    cache.join("data").join(index.to_string()),
                    cancel.clone(),
                    info_span!("prefetcher", run = %id, stream = %name),
                    name.clone(),
                );
                (name, prefetch)
            })
            .collect();
        Ok(Self {
            cancel,
            streams,
            cache,
        })
    }
    pub fn start(
        mut self,
    ) -> (
        HashMap<String, mpsc::Sender<BatchRequest>>,
        tokio::task::JoinHandle<()>,
    ) {
        let mut senders = HashMap::new();
        let mut workers = tokio::task::JoinSet::new();
        for (name, mut prefetch) in self.streams.drain() {
            let (tx, mut rx) = mpsc::channel::<BatchRequest>(1);
            senders.insert(name, tx);
            let cancel = self.cancel.clone();
            workers.spawn(async move {
                loop {
                    let request = tokio::select! {
                        biased;
                        () = cancel.cancelled() => break,
                        request = rx.recv() => match request { Some(request) => request, None => break },
                    };
                    let result = tokio::select! {
                        biased;
                        () = cancel.cancelled() => break,
                        result = prefetch.next_batch() => result,
                    };
                    let failed = result.is_err();
                    let _ = request.reply.send(result);
                    if failed { break; }
                }
                prefetch.finish().await;
            });
        }
        let handle = tokio::spawn(async move {
            self.cancel.cancelled().await;
            while workers.join_next().await.is_some() {}
            if let Err(err) = fs::remove_dir_all(&self.cache).await
                && err.kind() != std::io::ErrorKind::NotFound
            {
                error!(error = %err, "removing run cache failed");
            }
        });
        (senders, handle)
    }
}
