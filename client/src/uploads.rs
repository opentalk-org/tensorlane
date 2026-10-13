use crate::{
    ipc::{Receiver, Sender},
    upload_client::{Reply, Upload},
};
use anyhow::{Context, ensure};
use reqwest::Method;
use sha2::{Digest, Sha256};
use std::{
    io::{Seek, SeekFrom},
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tensorlane_protocol::{
    ArtifactMetric, MetricBatch, SaveAssetMetadata, ScalarMetric, UploadMetadata, UploadSpec,
};
use tokio::{
    fs::File,
    io::{AsyncReadExt, AsyncSeekExt},
    net::{UnixListener, UnixStream},
    sync::{Semaphore, mpsc},
    task::JoinSet,
    time::{Duration, interval},
};
use tokio_util::sync::CancellationToken;

const CONCURRENT_ARTIFACTS: usize = 16;
const ARTIFACT_BACKLOG_WARNING: usize = 64;

pub async fn serve(
    listener: UnixListener,
    http: crate::transport::HttpClient,
    run_id: String,
    stopping: CancellationToken,
) -> anyhow::Result<()> {
    let mut connections = JoinSet::new();
    let mut failure = None;
    loop {
        tokio::select! {
            _ = stopping.cancelled() => break,
            accepted = listener.accept() => {
                let (socket, _) = accepted?;
                let http = http.clone();
                let run_id = run_id.clone();
                let stopping = stopping.clone();
                connections.spawn(async move {
                    let result = receive(socket, http, run_id, stopping).await;
                    if let Err(error) = &result {
                        eprintln!("TensorLane upload failed: {error:#}");
                    }
                    result
                });
            }
            Some(result) = connections.join_next() => {
                if let Err(error) = result? { failure = Some(error); }
            }
        }
    }
    drop(listener);
    while let Some(result) = connections.join_next().await {
        if let Err(error) = result? {
            failure = Some(error);
        }
    }
    failure.map_or(Ok(()), Err)
}

async fn receive(
    socket: UnixStream,
    http: crate::transport::HttpClient,
    run_id: String,
    stopping: CancellationToken,
) -> anyhow::Result<()> {
    let (read, write) = socket.into_split();
    let mut reader = Receiver::<Upload, _>::new(read);
    let mut replies = Sender::<Reply, _>::new(write);
    let (jobs, mut queue) = mpsc::channel(1024);
    let reading = async {
        loop {
            let message = tokio::select! {
                biased;
                message = reader.recv() => message?,
                _ = stopping.cancelled() => break,
            };
            let Some(message) = message else {
                break;
            };
            jobs.send(message).await?;
        }
        drop(jobs);
        anyhow::Ok(())
    };
    let mut automatic = false;
    let processing = async {
        let mut metrics = Metrics::new(http.clone(), run_id.clone());
        let permits = Arc::new(Semaphore::new(CONCURRENT_ARTIFACTS));
        let mut artifacts = JoinSet::new();
        let mut tick = interval(Duration::from_secs(1));
        let mut tick_failed = false;
        loop {
            let job = tokio::select! {
                job = queue.recv() => match job {
                    Some(job) => job,
                    None => break,
                },
                Some(result) = artifacts.join_next() => {
                    result??;
                    continue;
                }
                _ = tick.tick() => {
                    let flushed = metrics.flush().await;
                    if let Err(error) = &flushed && !tick_failed {
                        eprintln!("TensorLane metric upload failed, retrying: {error:#}");
                    }
                    tick_failed = flushed.is_err();
                    continue;
                }
            };
            match job {
                Upload::Automatic => automatic = true,
                Upload::SaveAsset {
                    asset_id,
                    name,
                    step,
                    path,
                    kind,
                    asset_type,
                    metadata_json,
                    content_type,
                } => {
                    save_asset(
                        http.clone(),
                        SaveAssetMetadata {
                            run_id: run_id.clone(),
                            asset_id,
                            name,
                            step,
                            kind,
                            asset_type,
                            metadata_json,
                            content_type,
                        },
                        path,
                    )
                    .await?
                }
                Upload::Flush => {
                    while let Some(result) = artifacts.join_next().await {
                        result??;
                    }
                    metrics.flush().await?;
                    replies.send(&Ok(())).await?;
                }
                Upload::Metric {
                    step,
                    name,
                    value,
                    timestamp_unix_ms,
                } => {
                    metrics
                        .metric(ScalarMetric {
                            step,
                            name,
                            value,
                            timestamp_unix_ms,
                        })
                        .await?;
                }
                Upload::Artifact {
                    step,
                    timestamp_unix_ms,
                    path,
                    name,
                    content_type,
                } => {
                    let http = http.clone();
                    let run_id = run_id.clone();
                    let permits = permits.clone();
                    artifacts.spawn(async move {
                        let _permit = permits.acquire().await?;
                        artifact(
                            http,
                            run_id,
                            ArtifactMetric {
                                step,
                                timestamp_unix_ms,
                                name,
                                size_bytes: 0,
                                content_type,
                            },
                            path,
                        )
                        .await
                    });
                    if artifacts.len() % ARTIFACT_BACKLOG_WARNING == 0 {
                        eprintln!(
                            "TensorLane artifact upload backlog: {} uploads pending; artifacts are logged faster than they upload",
                            artifacts.len()
                        );
                    }
                }
            }
        }
        while let Some(result) = artifacts.join_next().await {
            result??;
        }
        metrics.flush().await
    };
    let result = tokio::try_join!(reading, processing).map(|_| ());
    if let Err(error) = &result {
        let _ = replies.send(&Err(format!("{error:#}"))).await;
    }
    if automatic {
        if let Err(error) = &result {
            eprintln!("TensorLane performance metrics failed: {error:#}");
        }
        Ok(())
    } else {
        result
    }
}

pub(crate) fn timestamp() -> anyhow::Result<i64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}

struct Metrics {
    client: crate::transport::HttpClient,
    run_id: String,
    batch: MetricBatch,
    pending: Option<(String, MetricBatch)>,
}

impl Metrics {
    fn new(client: crate::transport::HttpClient, run_id: String) -> Self {
        Self {
            client,
            run_id,
            batch: MetricBatch::default(),
            pending: None,
        }
    }

    async fn metric(&mut self, metric: ScalarMetric) -> anyhow::Result<()> {
        self.batch.scalars.push(metric);
        if self.batch.scalars.len() >= 500 {
            self.flush().await?;
        }
        Ok(())
    }

    async fn flush(&mut self) -> anyhow::Result<()> {
        self.send_pending().await?;
        let batch = std::mem::take(&mut self.batch);
        let filled = !batch.scalars.is_empty() || !batch.arrays.is_empty();
        self.pending = filled.then(|| (uuid::Uuid::new_v4().to_string(), batch));
        self.send_pending().await
    }

    async fn send_pending(&mut self) -> anyhow::Result<()> {
        let Some((request, batch)) = &self.pending else {
            return Ok(());
        };
        self.client
            .request(
                Method::PUT,
                &["runs", &self.run_id, "metrics", request],
                Some(serde_json::to_vec(batch)?),
                &[("content-type", "application/json".into())],
            )
            .await?;
        self.pending = None;
        Ok(())
    }
}

async fn artifact(
    client: crate::transport::HttpClient,
    run_id: String,
    mut metric: ArtifactMetric,
    path: PathBuf,
) -> anyhow::Result<()> {
    let (file, directory) = upload_source(path).await?;
    metric.size_bytes = file.metadata().await?.len();
    if directory {
        metric.content_type = "application/x-tar".into();
    }
    let metadata = UploadMetadata::Artifact {
        run_id,
        metadata: metric,
    };
    transfer(&client, &uuid::Uuid::new_v4().to_string(), metadata, file).await?;
    Ok(())
}

async fn save_asset(
    client: crate::transport::HttpClient,
    mut metadata: SaveAssetMetadata,
    path: PathBuf,
) -> anyhow::Result<()> {
    let id = metadata.asset_id.clone();
    let (file, directory) = upload_source(path).await?;
    if directory {
        metadata.content_type = "application/x-tar".into();
    }
    let (file, sha256) = transfer(&client, &id, UploadMetadata::Asset { metadata }, file).await?;
    let file = file.into_std().await;
    let stored =
        tokio::task::spawn_blocking(move || crate::assets::store(&id, &sha256, file)).await?;
    if let Err(error) = stored {
        eprintln!("TensorLane could not cache uploaded asset: {error:#}");
    }
    Ok(())
}

async fn transfer(
    client: &crate::transport::HttpClient,
    id: &str,
    metadata: UploadMetadata,
    mut file: File,
) -> anyhow::Result<(File, String)> {
    let size = file.metadata().await?.len();
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 256 * 1024];
    loop {
        let count = file.read(&mut buffer).await?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    file.seek(SeekFrom::Start(0)).await?;
    let spec = UploadSpec {
        metadata,
        size,
        sha256: hex::encode(hash.finalize()),
    };
    let status = client.upload(id, &spec, &file).await?;
    ensure!(status.committed, "upload was not committed");
    Ok((file, spec.sha256))
}

async fn upload_source(path: PathBuf) -> anyhow::Result<(File, bool)> {
    let (file, is_directory) = tokio::task::spawn_blocking(move || {
        let metadata = std::fs::symlink_metadata(&path).context("opening asset")?;
        ensure!(
            metadata.is_file() || metadata.is_dir(),
            "upload path must be a file or directory"
        );
        if metadata.is_file() {
            return anyhow::Ok((std::fs::File::open(path)?, false));
        }
        let name = path.file_name().context("upload path must have a name")?;
        let mut archive = tar::Builder::new(tempfile::tempfile()?);
        archive.follow_symlinks(false);
        archive.append_dir_all(name, &path)?;
        let mut file = archive.into_inner()?;
        file.seek(SeekFrom::Start(0))?;
        anyhow::Ok((file, true))
    })
    .await
    .context("preparing upload failed")??;
    Ok((File::from_std(file), is_directory))
}
