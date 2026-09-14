use async_trait::async_trait;
use aws_sdk_s3::error::SdkError;
use bytes::{BufMut, Bytes, BytesMut};
use tracing::{debug, trace, warn};

use crate::{audio, sampling};

#[async_trait]
pub trait Loader: Send + Sync {
    async fn load(&self, sample: &sampling::Sample) -> anyhow::Result<Option<Bytes>>;
    async fn load_batch(
        &self,
        batch: Vec<sampling::Sample>,
    ) -> anyhow::Result<Option<Vec<(sampling::Sample, Bytes)>>> {
        debug!(samples = batch.len(), "loading batch");

        let res = futures::future::try_join_all(batch.into_iter().map(|sample| async move {
            let wave = self.load(&sample).await?;
            anyhow::Ok((sample, wave))
        }))
        .await?;

        let mut vec = vec![];
        for sample in res {
            match sample.1 {
                Some(data) => vec.push((sample.0, data)),
                None => return Ok(None),
            }
        }

        return Ok(Some(vec));
    }
}

#[derive(Clone)]
pub struct S3Loader {
    s3_client: aws_sdk_s3::Client,
    bucket: &'static str,
}

impl S3Loader {
    pub fn new(s3_client: aws_sdk_s3::Client, bucket: &'static str) -> Self {
        Self { s3_client, bucket }
    }
}

#[async_trait]
impl Loader for S3Loader {
    async fn load(&self, sample: &sampling::Sample) -> anyhow::Result<Option<Bytes>> {
        trace!(
            audio = %sample.audio_id,
            object = %sample.object.path,
            offset = sample.object.offset,
            length = sample.object.length,
            "fetching audio from bucket"
        );
        let obj = match self
            .s3_client
            .get_object()
            .bucket(self.bucket)
            .key(&sample.object.path)
            .range(format!(
                "bytes={}-{}",
                sample.object.offset,
                sample.object.offset + sample.object.length - 1
            ))
            .send()
            .await
        {
            Err(SdkError::TimeoutError(err)) => {
                warn!(error = ?err, "s3 request timeout, skipping sample");
                return Ok(None);
            }
            Err(SdkError::ResponseError(err)) if err.raw().status().is_server_error() => {
                warn!(error = ?err, "s3 internal server error, skipping sample");
                return Ok(None);
            }
            Err(SdkError::ServiceError(err)) if err.raw().status().is_server_error() => {
                warn!(error = ?err, "s3 internal server error, skipping sample");
                return Ok(None);
            }
            other => other?,
        };

        let mut stream = obj.body;

        let mut buff = BytesMut::new();
        while let Some(bytes) = stream.try_next().await? {
            buff.put(bytes);
        }
        let wave = audio::process_audio(buff.freeze(), 24_000)?;

        Ok(Some(wave))
    }
}
