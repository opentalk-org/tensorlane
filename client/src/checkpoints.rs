use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::Context;
use bytes::Bytes;
use flume::Receiver;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::proto::tensor_lane_client::TensorLaneClient;
use crate::proto::{CheckpointMetadata, CheckpointRequest, checkpoint_request};

const CHUNK_BYTES: usize = 2 * 1024 * 1024;

pub struct CheckpointJob {
    pub step: u64,
    pub source: PathBuf,
}

pub async fn worker(
    mut client: TensorLaneClient<tonic::transport::Channel>,
    training_id: String,
    jobs: Receiver<CheckpointJob>,
) -> anyhow::Result<()> {
    while let Ok(job) = jobs.recv_async().await {
        upload(&mut client, &training_id, job).await?;
    }
    Ok(())
}

async fn upload(
    client: &mut TensorLaneClient<tonic::transport::Channel>,
    run_id: &str,
    job: CheckpointJob,
) -> anyhow::Result<()> {
    let (requests, receiver) = mpsc::channel(4);
    requests
        .send(CheckpointRequest {
            payload: Some(checkpoint_request::Payload::Metadata(CheckpointMetadata {
                run_id: run_id.to_owned(),
                step: job.step,
            })),
        })
        .await?;
    let producer = tokio::task::spawn_blocking(move || stream_tar(&job.source, requests));
    let rpc_result = client.checkpoint(ReceiverStream::new(receiver)).await;
    producer
        .await
        .context("joining checkpoint archive producer")??;
    rpc_result?;
    Ok(())
}

fn stream_tar(source: &Path, sender: mpsc::Sender<CheckpointRequest>) -> anyhow::Result<()> {
    let writer = ChunkWriter::new(sender);
    let mut archive = tar::Builder::new(writer);
    let mut entries = std::fs::read_dir(source)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let name = entry.file_name();
        if path.is_dir() {
            archive.append_dir_all(name, path)?;
        } else {
            archive.append_path_with_name(path, name)?;
        }
    }
    let writer = archive.into_inner()?;
    writer.finish()
}

struct ChunkWriter {
    sender: mpsc::Sender<CheckpointRequest>,
    buffer: Vec<u8>,
}

impl ChunkWriter {
    fn new(sender: mpsc::Sender<CheckpointRequest>) -> Self {
        Self {
            sender,
            buffer: Vec::with_capacity(CHUNK_BYTES),
        }
    }

    fn finish(mut self) -> anyhow::Result<()> {
        self.send_buffer()?;
        Ok(())
    }

    fn send_buffer(&mut self) -> io::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let bytes = Bytes::from(std::mem::take(&mut self.buffer));
        self.buffer = Vec::with_capacity(CHUNK_BYTES);
        self.sender
            .blocking_send(CheckpointRequest {
                payload: Some(checkpoint_request::Payload::Chunk(bytes)),
            })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "checkpoint RPC ended"))
    }
}

impl Write for ChunkWriter {
    fn write(&mut self, mut bytes: &[u8]) -> io::Result<usize> {
        let original = bytes.len();
        while !bytes.is_empty() {
            let available = CHUNK_BYTES - self.buffer.len();
            let take = available.min(bytes.len());
            self.buffer.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];
            if self.buffer.len() == CHUNK_BYTES {
                self.send_buffer()?;
            }
        }
        Ok(original)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.send_buffer()
    }
}
