use crate::{MAX_BATCH_BYTES, sampling::BlobRef};
use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use aws_sdk_s3::Client;
use tokio::io::AsyncReadExt;

#[async_trait]
pub trait Loader: Send + Sync {
    async fn size(&self, reference: &BlobRef) -> Result<usize>;
    async fn load(&self, reference: &BlobRef, size: usize) -> Result<Vec<u8>>;
}
#[derive(Clone)]
pub struct S3Loader {
    s3_client: Client,
    bucket: &'static str,
}
impl S3Loader {
    pub fn new(s3_client: Client, bucket: &'static str) -> Self {
        Self { s3_client, bucket }
    }
}
#[async_trait]
impl Loader for S3Loader {
    async fn size(&self, reference: &BlobRef) -> Result<usize> {
        let size = match reference.byte_length {
            Some(size) => size,
            None => {
                let head = self
                    .s3_client
                    .head_object()
                    .bucket(self.bucket)
                    .key(&reference.object)
                    .send()
                    .await
                    .with_context(|| format!("inspecting blob {}", reference.object))?;
                u64::try_from(head.content_length().context("blob has no size")?)?
            }
        };
        ensure!(
            size <= MAX_BATCH_BYTES as u64,
            "blob exceeds the 64 MiB batch limit"
        );
        Ok(size as usize)
    }

    async fn load(&self, reference: &BlobRef, size: usize) -> Result<Vec<u8>> {
        let mut request = self
            .s3_client
            .get_object()
            .bucket(self.bucket)
            .key(&reference.object);
        if let (Some(offset), Some(length)) = (reference.byte_offset, reference.byte_length) {
            request = request.range(format!("bytes={offset}-{}", offset + length - 1));
        }
        let object = request
            .send()
            .await
            .with_context(|| format!("loading blob {}", reference.object))?;
        if let Some(length) = object.content_length() {
            ensure!(length == size as i64, "blob returned an unexpected size");
        }
        let mut stream = object.body.into_async_read();
        let mut bytes = vec![0; size];
        let mut offset = 0;
        while offset < size {
            let count = tokio::time::timeout(
                std::time::Duration::from_secs(30),
                stream.read(&mut bytes[offset..]),
            )
            .await??;
            ensure!(count > 0, "blob returned an unexpected size");
            offset += count;
        }
        let extra = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            stream.read(&mut [0u8; 1]),
        )
        .await??;
        ensure!(extra == 0, "blob returned an unexpected size");
        Ok(bytes)
    }
}
