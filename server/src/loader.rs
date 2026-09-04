use async_trait::async_trait;
use bytes::{BufMut, Bytes, BytesMut};
use tracing::{debug, trace};

use crate::{audio, sampling};

#[async_trait]
pub trait Loader: Send + Sync {
    async fn load(&self, sample: &sampling::Sample) -> anyhow::Result<Bytes>;
    async fn load_batch(
        &self,
        batch: Vec<sampling::Sample>,
    ) -> anyhow::Result<Vec<(sampling::Sample, Bytes)>> {
        debug!(samples = batch.len(), "loading batch");

        futures::future::try_join_all(batch.into_iter().map(|sample| async move {
            let wave = self.load(&sample).await?;
            anyhow::Ok((sample, wave))
        }))
        .await
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
    async fn load(&self, sample: &sampling::Sample) -> anyhow::Result<Bytes> {
        trace!(
            audio = %sample.audio_id,
            object = %sample.object.path,
            offset = sample.object.offset,
            length = sample.object.length,
            "fetching audio from bucket"
        );
        let obj = self
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
            .await?;

        let mut stream = obj.body;

        let mut buff = BytesMut::new();
        while let Some(bytes) = stream.try_next().await? {
            buff.put(bytes);
        }
        let wave = audio::process_audio(buff.freeze(), 24_000)?;

        Ok(wave)
    }
}

pub struct SyntheticLoader;

#[async_trait]
impl Loader for SyntheticLoader {
    async fn load(&self, sample: &sampling::Sample) -> anyhow::Result<Bytes> {
        let sample_count = (sample.duration * 24_000.0) as usize;
        let mut wave = BytesMut::with_capacity(2 * sample_count);
        for t in 0..sample_count {
            let phase = 2.0 * std::f64::consts::PI * 220.0 * t as f64 / 24_000.0;
            wave.put_i16_le((0.25 * phase.sin() * 32_767.0) as i16);
        }

        Ok(wave.freeze())
    }
}
