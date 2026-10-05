use crate::{
    MAX_BATCH_BYTES,
    sampling::{BlobRef, Sample},
};
use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use aws_sdk_s3::Client;
use bytes::{Bytes, BytesMut};
use futures::{StreamExt, TryStreamExt};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::Semaphore;

#[async_trait]
pub trait Loader: Send + Sync {
    async fn load(&self, reference: &BlobRef) -> Result<Bytes>;
    async fn load_sample(&self, sample: Sample) -> Result<tensorlane_protocol::Sample> {
        let mut reads = futures::stream::iter(sample.blobs.into_iter().map(
            |(name, reference)| async move {
                Ok::<_, anyhow::Error>((name, self.load(&reference).await?))
            },
        ))
        .buffer_unordered(2);
        let mut size = sample.sample_id.len() + sample.metadata_json.len();
        let mut blobs = HashMap::new();
        while let Some((name, bytes)) = reads.try_next().await? {
            size = size
                .checked_add(name.len())
                .and_then(|size| size.checked_add(bytes.len()))
                .context("sample size overflow")?;
            ensure!(
                size <= MAX_BATCH_BYTES,
                "sample exceeds the 64 MiB batch limit"
            );
            blobs.insert(name, bytes.to_vec());
        }
        Ok(tensorlane_protocol::Sample {
            sample_id: sample.sample_id,
            metadata_json: sample.metadata_json,
            blobs,
        })
    }
}
#[derive(Clone)]
pub struct S3Loader {
    s3_client: Client,
    bucket: &'static str,
    slots: Arc<Semaphore>,
}
impl S3Loader {
    pub fn new(s3_client: Client, bucket: &'static str) -> Self {
        Self {
            s3_client,
            bucket,
            slots: Arc::new(Semaphore::new(4)),
        }
    }
}
#[async_trait]
impl Loader for S3Loader {
    async fn load(&self, reference: &BlobRef) -> Result<Bytes> {
        let _permit = self.slots.acquire().await?;
        let mut request = self
            .s3_client
            .get_object()
            .bucket(self.bucket)
            .key(&reference.object);
        if let (Some(offset), Some(length)) = (reference.byte_offset, reference.byte_length) {
            ensure!(
                length <= MAX_BATCH_BYTES as u64,
                "blob exceeds the 64 MiB batch limit"
            );
            request = request.range(format!("bytes={offset}-{}", offset + length - 1));
        }
        let object = request
            .send()
            .await
            .with_context(|| format!("loading blob {}", reference.object))?;
        if let Some(length) = object.content_length() {
            ensure!(
                length >= 0 && length as usize <= MAX_BATCH_BYTES,
                "blob exceeds the 64 MiB batch limit"
            );
        }
        let mut stream = object.body;
        let mut bytes = BytesMut::new();
        while let Some(chunk) =
            tokio::time::timeout(std::time::Duration::from_secs(30), stream.try_next()).await??
        {
            ensure!(
                bytes.len() + chunk.len() <= MAX_BATCH_BYTES,
                "blob exceeds the 64 MiB batch limit"
            );
            bytes.extend_from_slice(&chunk);
        }
        if let Some(length) = reference.byte_length {
            ensure!(
                bytes.len() as u64 == length,
                "blob range returned an unexpected byte length"
            );
        }
        Ok(bytes.freeze())
    }
}
