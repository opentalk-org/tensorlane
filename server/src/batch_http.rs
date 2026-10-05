use crate::{
    MAX_BATCH_BYTES,
    db::stream_samples,
    loader::Loader,
    runtime::{Batch, Runtime},
    sampling::{BatchPlan, QuerySampler},
    shared_cache::{Lock, write_atomic},
};
use anyhow::{Context, Result, ensure};
use futures::{StreamExt, TryStreamExt};
use prost::Message;
use tensorlane_protocol::DataResponse;
use tokio::fs;
use uuid::Uuid;

impl Runtime {
    pub async fn batch(&self, id: Uuid, session: Uuid, name: &str, sequence: u64) -> Result<Batch> {
        ensure!(!self.shutdown.is_cancelled(), "server is shutting down");
        let (_, config) = self.active(id, session).await?;
        let (index, (_, query)) = config
            .queries
            .iter()
            .enumerate()
            .find(|(_, (key, _))| key.as_str() == name)
            .context("unknown stream")?;
        let dir = self.run_dir(id).join("plans");
        fs::create_dir_all(&dir).await?;
        let path = dir.join(format!("{index}.plan"));
        crate::job::check(&path.with_extension("error")).await?;
        if !fs::try_exists(path.with_extension("ready")).await? {
            let Some(lock) = Lock::try_acquire(&path.with_extension("lock")).await? else {
                return Ok(Batch::Pending);
            };
            let Ok(slot) = self.plans.clone().try_acquire_owned() else {
                return Ok(Batch::Pending);
            };
            if !fs::try_exists(path.with_extension("ready")).await? {
                let engine = self.clone();
                let ready = path.with_extension("ready");
                let path = path.clone();
                let query = query.clone();
                let name = name.to_owned();
                self.tasks.spawn(async move {
                    let (_lock, _slot) = (lock, slot);
                    let result = async {
                        crate::cache_limits::space(&engine.cache, 512*1024*1024).await?;
                        let _ = fs::remove_file(path.with_extension("part")).await;
                        let rows = stream_samples(&engine.database, &query.sql, &query.params);
                        let mut sampler = QuerySampler::create(&name, rows, &path, query.repeat).await?;
                        sampler.persist();
                        write_atomic(&path.with_extension("ready"), b"ready").await?;
                        anyhow::Ok(())
                    }.await;
                    if let Err(error) = result {
                        tracing::warn!(run = %id, stream = %name, error = %error, "query plan preparation failed");
                        crate::job::failed(&path.with_extension("error"), &error).await;
                    }
                });
                if !crate::job::wait_for_file(&ready).await? {
                    return Ok(Batch::Pending);
                }
            }
        }
        let mut sampler = QuerySampler::open(&path, query.repeat).await?;
        let Some(plan) = sampler.batch_at(sequence).await? else {
            return Ok(Batch::End);
        };
        let data = self.run_dir(id).join("data").join(index.to_string());
        fs::create_dir_all(&data).await?;
        let cached = data.join(format!("{}.batch", plan.query_batch_idx));
        crate::job::check(&cached.with_extension("error")).await?;
        if fs::try_exists(&cached).await? {
            return Ok(Batch::Ready(cached));
        }
        let Some(lock) = Lock::try_acquire(&cached.with_extension("lock")).await? else {
            return Ok(Batch::Pending);
        };
        let Ok(slot) = self.batches.clone().try_acquire_owned() else {
            return Ok(Batch::Pending);
        };
        let engine = self.clone();
        let result_path = cached.clone();
        let name = name.to_owned();
        self.tasks.spawn(async move {
            let (_lock, _slot) = (lock, slot);
            let result = async {
                let response = load_batch(engine.loader.as_ref(), name.clone(), plan).await?;
                write_atomic(&cached, &response.encode_to_vec()).await?;
                anyhow::Ok(())
            }.await;
            if let Err(error) = result {
                tracing::warn!(run = %id, stream = %name, error = %error, "batch preparation failed");
                crate::job::failed(&cached.with_extension("error"), &error).await;
            }
        });
        if crate::job::wait_for_file(&result_path).await? {
            Ok(Batch::Ready(result_path))
        } else {
            Ok(Batch::Pending)
        }
    }
}

async fn load_batch(loader: &dyn Loader, stream: String, plan: BatchPlan) -> Result<DataResponse> {
    let started = std::time::Instant::now();
    let mut response = DataResponse {
        stream,
        query_batch_idx: plan.query_batch_idx,
        ..Default::default()
    };
    let mut encoded_bytes = response.encoded_len();
    let mut samples = futures::stream::iter(plan.samples)
        .map(|sample| loader.load_sample(sample))
        .buffered(4);
    while let Some(sample) = samples.try_next().await? {
        encoded_bytes += prost::encoding::message::encoded_len(1, &sample);
        ensure!(
            encoded_bytes <= MAX_BATCH_BYTES - 64,
            "encoded batch exceeds 64 MiB"
        );
        response.batch.push(sample);
    }
    response.load_seconds = started.elapsed().as_secs_f64();
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::{BlobRef, Sample};
    use async_trait::async_trait;
    use bytes::Bytes;
    use std::{
        collections::BTreeMap,
        sync::{
            Mutex,
            atomic::{AtomicUsize, Ordering},
        },
        time::Duration,
    };

    struct DelayedLoader {
        barrier: tokio::sync::Barrier,
        active: AtomicUsize,
        peak: AtomicUsize,
        completed: Mutex<Vec<u8>>,
    }

    #[async_trait]
    impl Loader for DelayedLoader {
        async fn load(&self, reference: &BlobRef) -> Result<Bytes> {
            let index: u8 = reference.object.parse()?;
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(active, Ordering::SeqCst);
            self.barrier.wait().await;
            tokio::time::sleep(Duration::from_millis((8 - index as u64) * 5)).await;
            self.completed.lock().unwrap().push(index);
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(Bytes::from(vec![index]))
        }
    }

    fn plan(count: u8) -> BatchPlan {
        BatchPlan {
            query_batch_idx: 42,
            samples: (0..count)
                .map(|index| Sample {
                    sample_id: index.to_string(),
                    metadata_json: "{}".into(),
                    blobs: BTreeMap::from([(
                        "payload".into(),
                        BlobRef {
                            object: index.to_string(),
                            byte_offset: None,
                            byte_length: None,
                        },
                    )]),
                })
                .collect(),
        }
    }

    #[tokio::test]
    async fn concurrent_sample_reads_preserve_saved_order_and_replay() -> Result<()> {
        let loader = DelayedLoader {
            barrier: tokio::sync::Barrier::new(4),
            active: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            completed: Mutex::new(Vec::new()),
        };
        for _ in 0..2 {
            let batch = tokio::time::timeout(
                Duration::from_secs(2),
                load_batch(&loader, "training".into(), plan(8)),
            )
            .await??;
            assert_eq!(batch.query_batch_idx, 42);
            for (index, sample) in batch.batch.iter().enumerate() {
                assert_eq!(sample.sample_id, index.to_string());
                assert_eq!(sample.blobs["payload"], vec![index as u8]);
            }
        }
        assert_eq!(loader.peak.load(Ordering::SeqCst), 4);
        assert_ne!(loader.completed.lock().unwrap()[0], 0);
        Ok(())
    }

    struct LargeLoader;

    #[async_trait]
    impl Loader for LargeLoader {
        async fn load(&self, _: &BlobRef) -> Result<Bytes> {
            Ok(Bytes::from(vec![0; MAX_BATCH_BYTES / 2]))
        }
    }

    #[tokio::test]
    async fn concurrent_sample_reads_enforce_encoded_batch_limit() {
        let error = load_batch(&LargeLoader, "training".into(), plan(2))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("encoded batch exceeds 64 MiB"));
    }
}
