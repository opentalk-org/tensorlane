use crate::{
    MAX_BATCH_BYTES,
    db::stream_samples,
    loader::Loader,
    runtime::{Batch, LOAD_MEMORY_UNIT, Runtime},
    sampling::{BatchPlan, plan::QuerySampler},
    shared_cache::{Lock, write_atomic},
};
use anyhow::{Context, Result, ensure};
use futures::future::try_join_all;
use prost::Message;
use std::sync::Arc;
use tensorlane_protocol::DataResponse;
use tokio::{
    fs,
    sync::{OwnedSemaphorePermit, Semaphore},
};
use uuid::Uuid;

impl Runtime {
    pub async fn batch(&self, id: Uuid, session: Uuid, name: &str, sequence: u64) -> Result<Batch> {
        ensure!(!self.shutdown.is_cancelled(), "server is shutting down");
        let config = self.active(id, session).await?;
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
            if !fs::try_exists(path.with_extension("ready")).await? {
                let engine = self.clone();
                let ready = path.with_extension("ready");
                let path = path.clone();
                let query = query.clone();
                let name = name.to_owned();
                self.tasks.spawn(async move {
                    let _lock = lock;
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
        let limit = (config.max_load_memory_bytes / config.queries.len()).max(1);
        let memory = self.loading_memory(id, name, limit)?;
        let engine = self.clone();
        let result_path = cached.clone();
        let name = name.to_owned();
        self.tasks.spawn(async move {
            let _lock = lock;
            let result = async {
                let Some((response, _memory)) = load_batch(engine.loader.as_ref(), name.clone(), plan, memory, limit).await? else {
                    return Ok(());
                };
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

async fn load_batch(
    loader: &dyn Loader,
    stream: String,
    plan: BatchPlan,
    memory: Arc<Semaphore>,
    limit: usize,
) -> Result<Option<(DataResponse, OwnedSemaphorePermit)>> {
    let started = std::time::Instant::now();
    let mut response = DataResponse {
        stream,
        query_batch_idx: plan.query_batch_idx,
        ..Default::default()
    };
    let prepared = try_join_all(plan.samples.into_iter().map(|sample| async move {
        let sizes = try_join_all(sample.blobs.values().map(|blob| loader.size(blob))).await?;
        anyhow::Ok((sample, sizes))
    }))
    .await?;
    let mut payload_bytes = 0usize;
    let mut encoded_bound = response.encoded_len() + 64;
    let mut overhead = 0usize;
    for (sample, sizes) in &prepared {
        encoded_bound += sample.sample_id.len() + sample.metadata_json.len() + 32;
        overhead += 256;
        for ((name, blob), size) in sample.blobs.iter().zip(sizes) {
            payload_bytes += size;
            encoded_bound += name.len() + size + 32;
            // Descriptor/future storage and an allowance for each active S3 stream.
            overhead += blob.object.len() + name.len() + 64 * 1024;
        }
    }
    ensure!(
        payload_bytes <= MAX_BATCH_BYTES,
        "encoded batch exceeds 64 MiB"
    );
    let required = encoded_bound
        .checked_mul(2)
        .and_then(|size| size.checked_add(overhead))
        .context("batch loading memory size overflow")?;
    let count = u32::try_from(required.min(limit).div_ceil(LOAD_MEMORY_UNIT))?;
    // Oversized batches queue for the entire budget so smaller batches cannot
    // continually overtake them. They run alone once existing loads finish.
    let permit = if required > limit {
        memory.acquire_many_owned(count).await?
    } else {
        let Ok(permit) = memory.try_acquire_many_owned(count) else {
            return Ok(None);
        };
        permit
    };
    let mut encoded_bytes = response.encoded_len();
    let samples = try_join_all(prepared.into_iter().map(|(sample, sizes)| async move {
        let blobs = try_join_all(sample.blobs.into_iter().zip(sizes).map(
            |((name, blob), size)| async move {
                let bytes = loader.load(&blob, size).await?;
                ensure!(bytes.len() == size, "blob returned an unexpected size");
                anyhow::Ok((name, bytes))
            },
        ))
        .await?;
        anyhow::Ok(tensorlane_protocol::Sample {
            sample_id: sample.sample_id,
            metadata_json: sample.metadata_json,
            blobs: blobs.into_iter().collect(),
        })
    }))
    .await?;
    for sample in samples {
        encoded_bytes += prost::encoding::message::encoded_len(1, &sample);
        ensure!(
            encoded_bytes <= MAX_BATCH_BYTES - 64,
            "encoded batch exceeds 64 MiB"
        );
        response.batch.push(sample);
    }
    response.load_seconds = started.elapsed().as_secs_f64();
    Ok(Some((response, permit)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::{BlobRef, Sample};
    use async_trait::async_trait;
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
        async fn size(&self, _: &BlobRef) -> Result<usize> {
            Ok(1)
        }
        async fn load(&self, reference: &BlobRef, _: usize) -> Result<Vec<u8>> {
            let index: u8 = reference.object.parse()?;
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(active, Ordering::SeqCst);
            self.barrier.wait().await;
            tokio::time::sleep(Duration::from_millis((8 - index as u64) * 5)).await;
            self.completed.lock().unwrap().push(index);
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(vec![index])
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
                load_batch(
                    &loader,
                    "training".into(),
                    plan(8),
                    Arc::new(Semaphore::new(256 * 1024 * 1024 / LOAD_MEMORY_UNIT)),
                    256 * 1024 * 1024,
                ),
            )
            .await??
            .unwrap();
            let (batch, _permit) = batch;
            assert_eq!(batch.query_batch_idx, 42);
            for (index, sample) in batch.batch.iter().enumerate() {
                assert_eq!(sample.sample_id, index.to_string());
                assert_eq!(sample.blobs["payload"], vec![index as u8]);
            }
        }
        assert_eq!(loader.peak.load(Ordering::SeqCst), 8);
        assert_ne!(loader.completed.lock().unwrap()[0], 0);
        Ok(())
    }

    struct LargeLoader;

    #[tokio::test]
    async fn loading_waits_for_memory_without_fetching_and_releases_it_after_use() -> Result<()> {
        let limit = 256 * 1024 * 1024;
        let memory = Arc::new(Semaphore::new(limit / LOAD_MEMORY_UNIT));
        let occupied = memory
            .clone()
            .acquire_many_owned((limit / LOAD_MEMORY_UNIT) as u32)
            .await?;
        let loader = DelayedLoader {
            barrier: tokio::sync::Barrier::new(1),
            active: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            completed: Mutex::new(Vec::new()),
        };
        assert!(
            load_batch(&loader, "training".into(), plan(1), memory.clone(), limit)
                .await?
                .is_none()
        );
        assert_eq!(loader.peak.load(Ordering::SeqCst), 0);
        drop(occupied);
        let (response, permit) =
            load_batch(&loader, "training".into(), plan(1), memory.clone(), limit)
                .await?
                .unwrap();
        assert!(memory.available_permits() < limit / LOAD_MEMORY_UNIT);
        let _encoded = response.encode_to_vec();
        assert!(memory.available_permits() < limit / LOAD_MEMORY_UNIT);
        drop(permit);
        assert_eq!(memory.available_permits(), limit / LOAD_MEMORY_UNIT);
        Ok(())
    }

    #[tokio::test]
    async fn oversized_batch_runs_alone_without_starving_or_blocking_other_runs() -> Result<()> {
        let limit = 128 * 1024;
        let memory = Arc::new(Semaphore::new(limit / LOAD_MEMORY_UNIT));
        let loader = Arc::new(DelayedLoader {
            barrier: tokio::sync::Barrier::new(1),
            active: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            completed: Mutex::new(Vec::new()),
        });
        let occupied = memory.clone().acquire_owned().await?;
        let loading = {
            let memory = memory.clone();
            let loader = loader.clone();
            tokio::spawn(async move {
                load_batch(loader.as_ref(), "training".into(), plan(2), memory, limit).await
            })
        };
        tokio::task::yield_now().await;
        assert!(!loading.is_finished());
        assert_eq!(loader.peak.load(Ordering::SeqCst), 0);
        assert!(
            load_batch(
                loader.as_ref(),
                "training".into(),
                plan(1),
                memory.clone(),
                limit
            )
            .await?
            .is_none()
        );
        let other_run = Arc::new(Semaphore::new(limit / LOAD_MEMORY_UNIT));
        assert!(
            load_batch(
                loader.as_ref(),
                "training".into(),
                plan(1),
                other_run,
                limit
            )
            .await?
            .is_some()
        );
        drop(occupied);
        let (response, permit) = tokio::time::timeout(Duration::from_secs(1), loading)
            .await???
            .unwrap();
        assert_eq!(response.batch.len(), 2);
        assert_eq!(memory.available_permits(), 0);
        assert!(
            load_batch(
                loader.as_ref(),
                "training".into(),
                plan(1),
                memory.clone(),
                limit
            )
            .await?
            .is_none()
        );
        drop(permit);
        assert_eq!(memory.available_permits(), limit / LOAD_MEMORY_UNIT);
        assert!(
            load_batch(loader.as_ref(), "training".into(), plan(1), memory, limit)
                .await?
                .is_some()
        );
        Ok(())
    }

    #[async_trait]
    impl Loader for LargeLoader {
        async fn size(&self, _: &BlobRef) -> Result<usize> {
            Ok(MAX_BATCH_BYTES / 2)
        }
        async fn load(&self, _: &BlobRef, size: usize) -> Result<Vec<u8>> {
            Ok(vec![0; size])
        }
    }

    #[tokio::test]
    async fn concurrent_sample_reads_enforce_encoded_batch_limit() {
        let error = load_batch(
            &LargeLoader,
            "training".into(),
            plan(2),
            Arc::new(Semaphore::new(256 * 1024 * 1024 / LOAD_MEMORY_UNIT)),
            256 * 1024 * 1024,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("encoded batch exceeds 64 MiB"));
    }
}
