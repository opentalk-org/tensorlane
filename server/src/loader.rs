use crate::{MAX_BATCH_BYTES, sampling::BlobRef};
use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use aws_sdk_s3::Client;
use futures::future::try_join_all;
use tokio::io::AsyncReadExt;

#[async_trait]
pub trait Loader: Send + Sync {
    async fn size(&self, reference: &BlobRef) -> Result<usize>;
    async fn load(&self, reference: &BlobRef, size: usize) -> Result<Vec<u8>>;
}

struct BlobRead {
    reference: BlobRef,
    size: usize,
    parts: Vec<(usize, usize, usize)>,
}

pub async fn load_blobs(
    loader: &dyn Loader,
    references: Vec<(BlobRef, usize)>,
) -> Result<Vec<Vec<u8>>> {
    let mut references: Vec<_> = references.into_iter().enumerate().collect();
    references.sort_unstable_by(|(_, (a, _)), (_, (b, _))| {
        (&a.object, a.byte_offset).cmp(&(&b.object, b.byte_offset))
    });
    let mut reads = Vec::<BlobRead>::new();
    for (index, (reference, size)) in references {
        if let Some(previous) = reads.last_mut()
            && previous.reference.object == reference.object
            && let (Some(start), Some(count), Some(offset)) = (
                previous.reference.byte_offset,
                previous.reference.byte_length,
                reference.byte_offset,
            )
            && start.checked_add(count) == Some(offset)
            && reference.byte_length.is_some()
        {
            previous.parts.push((index, previous.size, size));
            previous.size += size;
            previous.reference.byte_length = Some(previous.size as u64);
            continue;
        }
        reads.push(BlobRead {
            reference,
            size,
            parts: vec![(index, 0, size)],
        });
    }
    let mut blobs: Vec<_> = try_join_all(reads.into_iter().map(
        |BlobRead {
             reference,
             size,
             parts,
         }| async move {
            let bytes = loader.load(&reference, size).await?;
            ensure!(bytes.len() == size, "blob returned an unexpected size");
            if parts.len() == 1 {
                return anyhow::Ok(vec![(parts[0].0, bytes)]);
            }
            Ok(parts
                .into_iter()
                .map(|(index, start, size)| (index, bytes[start..start + size].to_vec()))
                .collect())
        },
    ))
    .await?
    .into_iter()
    .flatten()
    .collect();
    blobs.sort_unstable_by_key(|(index, _)| *index);
    Ok(blobs.into_iter().map(|(_, bytes)| bytes).collect())
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct RecordingLoader(Mutex<Vec<(String, Option<u64>, usize)>>);

    #[async_trait]
    impl Loader for RecordingLoader {
        async fn size(&self, reference: &BlobRef) -> Result<usize> {
            Ok(reference.byte_length.unwrap_or(12) as usize)
        }
        async fn load(&self, reference: &BlobRef, size: usize) -> Result<Vec<u8>> {
            self.0
                .lock()
                .unwrap()
                .push((reference.object.clone(), reference.byte_offset, size));
            let start = reference.byte_offset.unwrap_or(0) as usize;
            Ok(b"0123456789ab"[start..start + size].to_vec())
        }
    }

    #[tokio::test]
    async fn adjacent_ranges_share_reads_and_preserve_input_order() -> Result<()> {
        let loader = RecordingLoader(Mutex::new(Vec::new()));
        let ranges = [
            ("a", Some(2), Some(4)),
            ("b", Some(2), Some(2)),
            ("a", Some(0), Some(2)),
            ("b", Some(0), Some(2)),
            ("a", Some(8), Some(1)),
            ("c", None, None),
        ];
        let references = ranges
            .into_iter()
            .map(|(object, byte_offset, byte_length)| {
                (
                    BlobRef {
                        object: object.into(),
                        byte_offset,
                        byte_length,
                    },
                    byte_length.unwrap_or(12) as usize,
                )
            })
            .collect();
        let blobs = load_blobs(&loader, references).await?;
        assert_eq!(
            blobs,
            vec![
                b"2345".to_vec(),
                b"23".to_vec(),
                b"01".to_vec(),
                b"01".to_vec(),
                b"8".to_vec(),
                b"0123456789ab".to_vec()
            ]
        );
        let mut reads = loader.0.lock().unwrap().clone();
        reads.sort();
        assert_eq!(
            reads,
            vec![
                ("a".into(), Some(0), 6),
                ("a".into(), Some(8), 1),
                ("b".into(), Some(0), 4),
                ("c".into(), None, 12)
            ]
        );
        Ok(())
    }
}
