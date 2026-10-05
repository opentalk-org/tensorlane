use anyhow::{Context, Result, ensure};
use axum::{
    body::Body,
    http::{StatusCode, header},
    response::Response,
};
use futures::TryStreamExt;
use sha2::{Digest, Sha256};
use tokio::{
    fs,
    io::{AsyncWriteExt, BufWriter},
};

use crate::{
    runtime::{AssetSource, Runtime},
    shared_cache::{Lock, TemporaryFile},
};
use tensorlane_protocol::TRANSFER_CHUNK_BYTES;

pub async fn download(engine: Runtime, source: AssetSource, range: (u64, u64)) -> Result<Response> {
    let permit = engine
        .asset_slots
        .clone()
        .try_acquire_owned()
        .context("asset download capacity reached")?;
    let size = source.download.size;
    let (start, end) = range;
    let identity = hex::encode(Sha256::digest(
        format!("{}\0{}", source.object, source.download.etag).as_bytes(),
    ));
    let dir = engine.cache.join("assets").join(identity);
    fs::create_dir_all(&dir).await?;
    let path = dir.join(format!("{start}-{end}"));
    let lock = Lock::acquire(&dir.join("asset.lock")).await?;
    let body = if let Ok(bytes) = fs::read(&path).await {
        ensure!(
            bytes.len() as u64 == end - start + 1,
            "invalid cached asset range"
        );
        Body::from(bytes)
    } else {
        let object = engine
            .s3
            .get_object()
            .bucket(engine.bucket)
            .key(&source.object)
            .if_match(&source.download.etag)
            .range(format!("bytes={start}-{end}"))
            .send()
            .await?;
        ensure!(
            object.content_length() == Some((end - start + 1) as i64),
            "R2 returned an unexpected asset range size"
        );
        crate::cache_limits::space(&dir, end - start + 1).await?;
        let part = TemporaryFile(path.with_extension(format!("{}.part", uuid::Uuid::new_v4())));
        let file = fs::File::create(&part.0).await?;
        Body::from_stream(async_stream::try_stream! {
            let (_lock, _permit, part) = (lock, permit, part);
            let mut file = BufWriter::with_capacity(256 * 1024, file);
            let mut upstream = object.body;
            let mut received = 0u64;
            while let Some(chunk) = tokio::time::timeout(std::time::Duration::from_secs(30), upstream.try_next()).await?? {
                received += chunk.len() as u64;
                if received > end - start + 1 { Err(anyhow::anyhow!("R2 sent more asset bytes than requested"))?; }
                file.write_all(&chunk).await?;
                if received == end - start + 1 {
                    file.flush().await?;
                    file.get_ref().sync_all().await?;
                    fs::rename(&part.0, &path).await?;
                }
                yield chunk;
            }
            if received != end - start + 1 { Err(anyhow::anyhow!("truncated R2 asset range"))?; }
            file.flush().await?;
            file.get_ref().sync_all().await?;
            drop(file);
        }.map_err(|e: anyhow::Error| std::io::Error::other(e.to_string())))
    };
    Ok(Response::builder()
        .status(StatusCode::PARTIAL_CONTENT)
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header(header::ACCEPT_RANGES, "bytes")
        .header(header::CONTENT_RANGE, format!("bytes {start}-{end}/{size}"))
        .header(header::CONTENT_LENGTH, end - start + 1)
        .header(header::ETAG, source.download.etag)
        .body(body)?)
}

pub fn parse_range(range: &str, size: u64) -> Result<(u64, u64)> {
    let (start, end) = range
        .strip_prefix("bytes=")
        .context("invalid asset range")?
        .split_once('-')
        .context("invalid asset range")?;
    let start: u64 = start.parse().context("invalid range start")?;
    let end: u64 = end.parse().context("invalid range end")?;
    ensure!(
        start <= end && end < size,
        "asset range is outside the object"
    );
    ensure!(
        end - start < TRANSFER_CHUNK_BYTES as u64,
        "asset ranges must not exceed 4 MiB"
    );
    Ok((start, end))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ranges_are_bounded_and_cannot_overflow() {
        assert_eq!(parse_range("bytes=1-3", 4).unwrap(), (1, 3));
        for range in [
            "bytes=3-4",
            "bytes=4-3",
            "bytes=0-",
            "bytes=-1",
            "bytes=0-1,3-4",
            "bytes=0-18446744073709551615",
        ] {
            assert!(parse_range(range, 4).is_err());
        }
        assert!(parse_range("bytes=0-4194304", 4194305).is_err());
    }
}
