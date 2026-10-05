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
    time::{SystemTime, UNIX_EPOCH},
};
use tensorlane_protocol::{
    ArtifactMetric, MetricBatch, SaveAssetMetadata, ScalarMetric, UploadMetadata, UploadSpec,
};
use tokio::{
    fs::File,
    io::{AsyncReadExt, AsyncSeekExt},
    net::{UnixListener, UnixStream},
    sync::mpsc,
    task::JoinSet,
};
use tokio_util::sync::CancellationToken;

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
        while let Some(job) = queue.recv().await {
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
                    metrics.flush().await?;
                    replies.send(&Ok(())).await?;
                }
                Upload::Metric { step, name, value } => {
                    metrics
                        .metric(ScalarMetric {
                            step,
                            name,
                            value,
                            timestamp_unix_ms: timestamp()?,
                        })
                        .await?;
                }
                Upload::Artifact {
                    step,
                    path,
                    name,
                    content_type,
                } => {
                    metrics.artifact(step, path, name, content_type).await?;
                }
            }
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

fn timestamp() -> anyhow::Result<i64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}

struct Metrics {
    client: crate::transport::HttpClient,
    run_id: String,
    batch: MetricBatch,
}

impl Metrics {
    fn new(client: crate::transport::HttpClient, run_id: String) -> Self {
        Self {
            client,
            run_id,
            batch: MetricBatch::default(),
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
        if self.batch.scalars.is_empty() && self.batch.arrays.is_empty() {
            return Ok(());
        }
        let request = uuid::Uuid::new_v4().to_string();
        self.client
            .request(
                Method::PUT,
                &["runs", &self.run_id, "metrics", &request],
                Some(serde_json::to_vec(&self.batch)?),
                &[("content-type", "application/json".into())],
                1024 * 1024,
            )
            .await?;
        self.batch = MetricBatch::default();
        Ok(())
    }

    async fn artifact(
        &mut self,
        step: u64,
        path: PathBuf,
        name: String,
        content_type: String,
    ) -> anyhow::Result<()> {
        self.flush().await?;
        let (file, directory) = upload_source(path).await?;
        let size = file.metadata().await?.len();
        let metadata = UploadMetadata::Artifact {
            run_id: self.run_id.clone(),
            metadata: ArtifactMetric {
                step,
                timestamp_unix_ms: timestamp()?,
                name,
                size_bytes: size,
                content_type: if directory {
                    "application/x-tar".into()
                } else {
                    content_type
                },
            },
        };
        transfer(
            &self.client,
            &uuid::Uuid::new_v4().to_string(),
            metadata,
            file,
        )
        .await
    }
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
    transfer(&client, &id, UploadMetadata::Asset { metadata }, file).await
}

async fn transfer(
    client: &crate::transport::HttpClient,
    id: &str,
    metadata: UploadMetadata,
    mut file: File,
) -> anyhow::Result<()> {
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
    Ok(())
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
