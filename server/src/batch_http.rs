use crate::{
    db::{SampleRow, stream_samples},
    loader::{Loader, load_blobs},
    runtime::{Batch, LOAD_MEMORY_UNIT, Runtime},
    sampling::{BatchPlan, QuerySampler},
    shared_cache::{Lock, write_atomic},
};
use anyhow::{Context, Result, ensure};
use futures::{Stream, future::try_join_all};
use prost::Message;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::{Instant, SystemTime},
};
use tensorlane_protocol::DataResponse;
use tokio::{
    fs,
    sync::{OwnedSemaphorePermit, Semaphore},
};
use uuid::Uuid;

impl Runtime {
    pub async fn batch(&self, id: Uuid, name: &str, sequence: u64) -> Result<Batch> {
        ensure!(!self.shutdown.is_cancelled(), "server is shutting down");
        let config = self.active(id).await?;
        let (index, (_, query)) = config
            .queries
            .iter()
            .enumerate()
            .find(|(_, (key, _))| key.as_str() == name)
            .context("unknown stream")?;
        let dir = self.run_dir(id).join("plans-v2");
        fs::create_dir_all(&dir).await?;
        let path = dir.join(format!("{index}.plan"));
        let ready = path.with_extension("ready");
        if !fs::try_exists(&ready).await? {
            let Some(lock) = Lock::try_acquire(&path.with_extension("lock")).await? else {
                return Ok(Batch::Pending);
            };
            crate::job::check(&path.with_extension("error")).await?;
            if !fs::try_exists(&ready).await? {
                let rows = stream_samples(&self.database, &query.sql, &query.params);
                self.tasks.spawn(prepare_plan(
                    self.budget.clone(),
                    name.to_owned(),
                    rows,
                    path.clone(),
                    lock,
                ));
            }
            if !crate::job::wait_for_file(&ready).await? {
                return Ok(Batch::Pending);
            }
        }
        let generation = Uuid::from_slice(&fs::read(&ready).await?)?;
        fs::File::open(&ready)
            .await?
            .into_std()
            .await
            .set_modified(SystemTime::now())?;
        let mut sampler = QuerySampler::open(&path, query.repeat).await?;
        let Some(plan) = sampler.batch_at(sequence).await? else {
            return Ok(Batch::End);
        };
        let data = self
            .run_dir(id)
            .join("data")
            .join(index.to_string())
            .join(generation.to_string());
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
        self.tasks.spawn(prepare_batch(
            self.clone(),
            name.to_owned(),
            plan,
            memory,
            limit,
            cached.clone(),
            lock,
        ));
        if crate::job::wait_for_file(&cached).await? {
            Ok(Batch::Ready(cached))
        } else {
            Ok(Batch::Pending)
        }
    }
}

async fn prepare_plan(
    budget: Arc<crate::cache_limits::Budget>,
    name: String,
    rows: impl Stream<Item = Result<SampleRow>> + Send,
    path: PathBuf,
    _lock: Lock,
) {
    if let Err(error) = QuerySampler::create(&budget, &name, rows, &path).await {
        crate::job::failed(&path.with_extension("error"), &error).await;
    }
}

async fn prepare_batch(
    engine: Runtime,
    stream: String,
    plan: BatchPlan,
    memory: Arc<Semaphore>,
    limit: usize,
    path: PathBuf,
    _lock: Lock,
) {
    if let Err(error) = write_batch(&engine, stream, plan, memory, limit, &path).await {
        crate::job::failed(&path.with_extension("error"), &error).await;
    }
}

async fn write_batch(
    engine: &Runtime,
    stream: String,
    plan: BatchPlan,
    memory: Arc<Semaphore>,
    limit: usize,
    path: &Path,
) -> Result<()> {
    let (response, _permit) =
        load_batch(engine.loader.as_ref(), stream, plan, memory, limit).await?;
    let bytes = response.encode_to_vec();
    let _reserved = engine.budget.reserve(bytes.len() as u64).await?;
    write_atomic(path, &bytes).await
}

async fn load_batch(
    loader: &dyn Loader,
    stream: String,
    plan: BatchPlan,
    memory: Arc<Semaphore>,
    limit: usize,
) -> Result<(DataResponse, OwnedSemaphorePermit)> {
    let started = Instant::now();
    let mut response = DataResponse {
        stream,
        query_batch_idx: plan.query_batch_idx,
        ..Default::default()
    };
    let references: Vec<_> = plan
        .samples
        .iter()
        .flat_map(|sample| sample.blobs.values().cloned())
        .collect();
    let sizes = try_join_all(references.iter().map(|blob| loader.size(blob))).await?;
    let mut size = sizes.iter();
    let mut encoded = response.encoded_len() + 64;
    let mut overhead = 0;
    for sample in &plan.samples {
        encoded += sample.sample_id.len() + sample.metadata_json.len() + 32;
        overhead += 256;
        for (name, blob) in &sample.blobs {
            encoded += name.len() + size.next().unwrap() + 32;
            overhead += blob.object.len() + name.len() + 64 * 1024;
        }
    }
    let required = encoded * 2 + overhead;
    let waiting = Instant::now();
    let permit = memory
        .acquire_many_owned(required.min(limit).div_ceil(LOAD_MEMORY_UNIT) as u32)
        .await?;
    response.server_wait_seconds = waiting.elapsed().as_secs_f64();
    let mut blobs = load_blobs(loader, references.into_iter().zip(sizes).collect())
        .await?
        .into_iter();
    for sample in plan.samples {
        response.batch.push(tensorlane_protocol::Sample {
            sample_id: sample.sample_id,
            metadata_json: sample.metadata_json,
            blobs: sample
                .blobs
                .into_keys()
                .map(|name| (name, blobs.next().unwrap()))
                .collect(),
        });
    }
    response.load_seconds = started.elapsed().as_secs_f64() - response.server_wait_seconds;
    Ok((response, permit))
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
            .await??;
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
        let loading = load_batch(&loader, "training".into(), plan(1), memory.clone(), limit);
        tokio::pin!(loading);
        assert!(futures::poll!(loading.as_mut()).is_pending());
        assert_eq!(loader.peak.load(Ordering::SeqCst), 0);
        drop(occupied);
        let (response, permit) = loading.await?;
        assert!(memory.available_permits() < limit / LOAD_MEMORY_UNIT);
        let _encoded = response.encode_to_vec();
        assert!(memory.available_permits() < limit / LOAD_MEMORY_UNIT);
        drop(permit);
        assert_eq!(memory.available_permits(), limit / LOAD_MEMORY_UNIT);
        Ok(())
    }

    #[tokio::test]
    async fn cancelled_waiting_batch_releases_partial_reservation() -> Result<()> {
        let limit = 16 * LOAD_MEMORY_UNIT;
        let memory = Arc::new(Semaphore::new(16));
        let occupied = memory.clone().acquire_many_owned(14).await?;
        let loader = DelayedLoader {
            barrier: tokio::sync::Barrier::new(1),
            active: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            completed: Mutex::new(Vec::new()),
        };
        {
            let waiting = load_batch(&loader, "audio".into(), plan(4), memory.clone(), limit);
            tokio::pin!(waiting);
            assert!(futures::poll!(waiting.as_mut()).is_pending());
            assert_eq!(loader.peak.load(Ordering::SeqCst), 0);
        }
        assert_eq!(memory.available_permits(), 2);
        drop(occupied);
        let (response, permit) = tokio::time::timeout(
            Duration::from_secs(1),
            load_batch(&loader, "audio".into(), plan(8), memory.clone(), limit),
        )
        .await??;
        assert_eq!(response.batch.len(), 8);
        drop(permit);
        assert_eq!(memory.available_permits(), 16);
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
        let smaller = load_batch(
            loader.as_ref(),
            "training".into(),
            plan(1),
            memory.clone(),
            limit,
        );
        tokio::pin!(smaller);
        assert!(futures::poll!(smaller.as_mut()).is_pending());
        let other_run = Arc::new(Semaphore::new(limit / LOAD_MEMORY_UNIT));
        let (other, _) = load_batch(
            loader.as_ref(),
            "training".into(),
            plan(1),
            other_run,
            limit,
        )
        .await?;
        assert_eq!(other.batch.len(), 1);
        drop(occupied);
        let (response, permit) = tokio::time::timeout(Duration::from_secs(1), loading).await???;
        assert_eq!(response.batch.len(), 2);
        assert_eq!(memory.available_permits(), 0);
        assert!(futures::poll!(smaller.as_mut()).is_pending());
        drop(permit);
        let (response, _) = tokio::time::timeout(Duration::from_secs(1), smaller).await??;
        assert_eq!(response.batch.len(), 1);
        assert_eq!(memory.available_permits(), limit / LOAD_MEMORY_UNIT);
        Ok(())
    }

    #[async_trait]
    impl Loader for LargeLoader {
        async fn size(&self, _: &BlobRef) -> Result<usize> {
            Ok(32 * 1024 * 1024)
        }
        async fn load(&self, _: &BlobRef, size: usize) -> Result<Vec<u8>> {
            Ok(vec![0; size])
        }
    }

    #[tokio::test]
    async fn concurrent_sample_reads_allow_large_batches() {
        let (response, _memory) = load_batch(
            &LargeLoader,
            "training".into(),
            plan(2),
            Arc::new(Semaphore::new(256 * 1024 * 1024 / LOAD_MEMORY_UNIT)),
            256 * 1024 * 1024,
        )
        .await
        .unwrap();
        assert_eq!(response.batch.len(), 2);
        assert!(response.encoded_len() > 64 * 1024 * 1024);
    }
}
