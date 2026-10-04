use crate::{
    MAX_BATCH_BYTES,
    db::stream_samples,
    runtime::{Batch, Runtime},
    sampling::QuerySampler,
    shared_cache::{Lock, write_atomic},
};
use anyhow::{Context, Result, ensure};
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
                let started = std::time::Instant::now();
                let mut response = DataResponse { stream: name.clone(), query_batch_idx: plan.query_batch_idx, ..Default::default() };
                for sample in plan.samples {
                    let sample = engine.loader.load_sample(sample).await?;
                    response.batch.push(sample);
                    ensure!(response.encoded_len() <= MAX_BATCH_BYTES - 64, "encoded batch exceeds 64 MiB");
                }
                response.load_seconds = started.elapsed().as_secs_f64();
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
