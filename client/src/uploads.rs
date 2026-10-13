use crate::{
    ipc::{Receiver, Sender},
    transport::HttpClient,
    upload_client::{Reply, Upload},
};
use anyhow::{Context, ensure};
use reqwest::Method;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs::File,
    io::{Seek, SeekFrom},
    path::PathBuf,
    sync::Arc,
};
use tensorlane_protocol::{
    ArtifactMetric, MetricBatch, SaveAssetMetadata, ScalarMetric, UploadMetadata, UploadSpec,
};
use tokio::{
    net::{UnixListener, UnixStream, unix::OwnedReadHalf, unix::OwnedWriteHalf},
    sync::{Semaphore, mpsc, oneshot},
    task::{JoinSet, spawn_blocking},
    time::{Duration, interval},
};
use tokio_util::sync::CancellationToken;

enum MetricJob {
    Batch(MetricBatch),
    Flush(oneshot::Sender<Vec<String>>),
}

pub async fn serve(
    listener: UnixListener,
    http: HttpClient,
    run_id: String,
    stopping: CancellationToken,
) -> anyhow::Result<()> {
    let mut connections = JoinSet::new();
    let mut failure = None;
    loop {
        tokio::select! {
            _ = stopping.cancelled() => break,
            accepted = listener.accept() => match accepted {
                Ok((socket, _)) => {
                    connections.spawn(receive(socket, http.clone(), run_id.clone(), stopping.clone()));
                }
                Err(error) => {
                    eprintln!("TensorLane upload connection failed: {error}");
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            },
            Some(result) = connections.join_next() => failure = result?.err().or(failure),
        }
    }
    drop(listener);
    while let Some(result) = connections.join_next().await {
        failure = result?.err().or(failure);
    }
    failure.map_or(Ok(()), Err)
}

async fn receive(
    socket: UnixStream,
    http: HttpClient,
    run_id: String,
    stopping: CancellationToken,
) -> anyhow::Result<()> {
    let (read, write) = socket.into_split();
    let mut replies = Sender::new(write);
    let mut automatic = false;
    let result = process(read, &mut replies, http, run_id, stopping, &mut automatic).await;
    let Err(error) = result else {
        return Ok(());
    };
    let _ = replies.send(&Err(format!("{error:#}"))).await;
    if automatic {
        eprintln!("TensorLane performance metrics failed: {error:#}");
        return Ok(());
    }
    eprintln!("TensorLane upload failed: {error:#}");
    Err(error)
}

async fn process(
    read: OwnedReadHalf,
    replies: &mut Sender<Reply, OwnedWriteHalf>,
    http: HttpClient,
    run_id: String,
    stopping: CancellationToken,
    automatic: &mut bool,
) -> anyhow::Result<()> {
    let mut reader = Receiver::<Upload, _>::new(read);
    let (batches, queue) = mpsc::unbounded_channel();
    let metrics_task = tokio::spawn(send_metrics(http.clone(), run_id.clone(), queue));
    let mut metrics = MetricBatch::default();
    let permits = Arc::new(Semaphore::new(16));
    let mut artifacts = JoinSet::new();
    let mut tick = interval(Duration::from_secs(1));
    let mut skipped = HashSet::new();
    let mut failures = Vec::new();
    let mut saved = oneshot::channel().1;
    loop {
        let job = tokio::select! {
            biased;
            message = reader.recv() => message?,
            Some(result) = artifacts.join_next() => {
                if let Err(error) = result? {
                    eprintln!("TensorLane {error:#}");
                    failures.push(format!("{error:#}"));
                }
                continue;
            }
            _ = tick.tick() => {
                if !metrics.scalars.is_empty() {
                    batches.send(MetricJob::Batch(std::mem::take(&mut metrics)))?;
                }
                continue;
            }
            _ = stopping.cancelled() => None,
        };
        match job {
            Some(Upload::Automatic) => *automatic = true,
            Some(Upload::Metric {
                step,
                name,
                value,
                timestamp_unix_ms,
            }) => {
                if !value.is_finite() || name.is_empty() {
                    if skipped.insert(name.clone()) {
                        eprintln!(
                            "TensorLane skipped metric {name:?} at step {step}: value {value} must be finite and name non-empty; later skips are not logged"
                        );
                    }
                    continue;
                }
                metrics.scalars.push(ScalarMetric {
                    step,
                    name,
                    value,
                    timestamp_unix_ms,
                });
                if metrics.scalars.len() >= 500 {
                    batches.send(MetricJob::Batch(std::mem::take(&mut metrics)))?;
                }
            }
            Some(Upload::Artifact {
                step,
                timestamp_unix_ms,
                path,
                name,
                content_type,
            }) => {
                let metric = ArtifactMetric {
                    step,
                    timestamp_unix_ms,
                    name,
                    size_bytes: 0,
                    content_type,
                };
                let task = artifact(http.clone(), run_id.clone(), permits.clone(), metric, path);
                artifacts.spawn(task);
                if artifacts.len() % 64 == 0 {
                    eprintln!(
                        "TensorLane upload backlog: {} uploads pending; files are saved faster than they upload",
                        artifacts.len()
                    );
                }
            }
            Some(Upload::SaveAsset {
                asset_id,
                name,
                step,
                path,
                kind,
                asset_type,
                metadata_json,
                content_type,
            }) => {
                let metadata = SaveAssetMetadata {
                    run_id: run_id.clone(),
                    asset_id,
                    name,
                    step,
                    kind,
                    asset_type,
                    metadata_json,
                    content_type,
                };
                let (done, next) = oneshot::channel();
                let previous = std::mem::replace(&mut saved, next);
                artifacts.spawn(save_asset(http.clone(), metadata, path, previous, done));
            }
            Some(Upload::Flush) | None => {
                while let Some(result) = artifacts.join_next().await {
                    if let Err(error) = result? {
                        eprintln!("TensorLane {error:#}");
                        failures.push(format!("{error:#}"));
                    }
                }
                batches.send(MetricJob::Batch(std::mem::take(&mut metrics)))?;
                let (done, flushed) = oneshot::channel();
                batches.send(MetricJob::Flush(done))?;
                failures.extend(flushed.await?);
                if job.is_none() {
                    break;
                }
                let reply = match failures.is_empty() {
                    true => Ok(()),
                    false => Err(failures.join("; ")),
                };
                failures.clear();
                replies.send(&reply).await?;
            }
        }
    }
    drop(batches);
    metrics_task.await??;
    ensure!(failures.is_empty(), "{}", failures.join("; "));
    Ok(())
}

async fn send_metrics(
    http: HttpClient,
    run_id: String,
    mut queue: mpsc::UnboundedReceiver<MetricJob>,
) -> anyhow::Result<()> {
    let mut failures = Vec::new();
    while let Some(job) = queue.recv().await {
        let batch = match job {
            MetricJob::Batch(batch) if batch.scalars.is_empty() => continue,
            MetricJob::Batch(batch) => batch,
            MetricJob::Flush(done) => {
                let _ = done.send(std::mem::take(&mut failures));
                continue;
            }
        };
        let request = uuid::Uuid::new_v4().to_string();
        let sent = http
            .request(
                Method::PUT,
                &["runs", &run_id, "metrics", &request],
                Some(serde_json::to_vec(&batch)?),
                &[("content-type", "application/json".into())],
            )
            .await;
        if let Err(error) = sent {
            let failure = format!("dropped {} metrics: {error:#}", batch.scalars.len());
            eprintln!("TensorLane {failure}");
            failures.push(failure);
        }
    }
    Ok(())
}

async fn artifact(
    http: HttpClient,
    run_id: String,
    permits: Arc<Semaphore>,
    mut metric: ArtifactMetric,
    path: PathBuf,
) -> anyhow::Result<()> {
    let _permit = permits.acquire().await?;
    let context = format!("uploading artifact {:?} failed", metric.name);
    let (file, directory, size, sha256) = spawn_blocking(move || prepare(path))
        .await?
        .context(context.clone())?;
    metric.size_bytes = size;
    if directory {
        metric.content_type = "application/x-tar".into();
    }
    let metadata = UploadMetadata::Artifact {
        run_id,
        metadata: metric,
    };
    let spec = UploadSpec {
        metadata,
        size,
        sha256,
    };
    let id = uuid::Uuid::new_v4().to_string();
    http.upload(&id, &spec, &file).await.context(context)
}

async fn save_asset(
    http: HttpClient,
    mut metadata: SaveAssetMetadata,
    path: PathBuf,
    previous: oneshot::Receiver<()>,
    _done: oneshot::Sender<()>,
) -> anyhow::Result<()> {
    let _ = previous.await;
    let context = format!("saving asset {:?} failed", metadata.name);
    let id = metadata.asset_id.clone();
    let (file, directory, size, sha256) = spawn_blocking(move || prepare(path))
        .await?
        .context(context.clone())?;
    if directory {
        metadata.content_type = "application/x-tar".into();
    }
    let spec = UploadSpec {
        metadata: UploadMetadata::Asset { metadata },
        size,
        sha256,
    };
    http.upload(&id, &spec, &file).await.context(context)?;
    let stored = spawn_blocking(move || crate::assets::store(&id, &spec.sha256, file)).await?;
    if let Err(error) = stored {
        eprintln!("TensorLane could not cache uploaded asset: {error:#}");
    }
    Ok(())
}

fn prepare(path: PathBuf) -> anyhow::Result<(File, bool, u64, String)> {
    let metadata = std::fs::symlink_metadata(&path).context("opening asset")?;
    ensure!(
        metadata.is_file() || metadata.is_dir(),
        "upload path must be a file or directory"
    );
    let mut file = if metadata.is_file() {
        File::open(&path)?
    } else {
        let name = path.file_name().context("upload path must have a name")?;
        let mut archive = tar::Builder::new(tempfile::tempfile()?);
        archive.follow_symlinks(false);
        archive.append_dir_all(name, &path)?;
        archive.into_inner()?
    };
    file.seek(SeekFrom::Start(0))?;
    let mut hash = Sha256::new();
    let size = std::io::copy(&mut file, &mut hash)?;
    Ok((file, metadata.is_dir(), size, hex::encode(hash.finalize())))
}
