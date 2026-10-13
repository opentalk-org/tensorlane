use anyhow::{Context, Result, ensure};
use axum::{
    body::Body,
    http::{StatusCode, header},
    response::Response,
};
use futures::TryStreamExt;
use sha2::{Digest, Sha256};
use std::path::Path;
use tokio::{
    fs,
    io::{AsyncWriteExt, BufWriter},
};

use crate::{
    runtime::{AssetSource, Runtime},
    shared_cache::{Lock, TemporaryFile},
};

pub async fn download(engine: Runtime, source: AssetSource, range: (u64, u64)) -> Result<Response> {
    let (start, end) = range;
    let length = end - start + 1;
    let identity = hex::encode(Sha256::digest(
        format!("{}\0{}", source.object, source.download.etag).as_bytes(),
    ));
    let dir = engine.cache.join("assets").join(identity);
    let path = dir.join(format!("{start}-{end}"));
    if let Some(body) = cached(&path).await? {
        return response(source, range, body);
    }
    fs::create_dir_all(&dir).await?;
    let lock = Lock::acquire(&path.with_extension("lock")).await?;
    if let Some(body) = cached(&path).await? {
        return response(source, range, body);
    }
    let permit = engine.asset_slots.clone().acquire_owned().await?;
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
        object.content_length() == Some(length as i64),
        "R2 returned an unexpected asset range size"
    );
    let reserved = engine.budget.reserve(length).await?;
    let part = TemporaryFile(path.with_extension(format!("{}.part", uuid::Uuid::new_v4())));
    let file = fs::File::create(&part.0).await?;
    let body = Body::from_stream(
        async_stream::try_stream! {
            let (_lock, _permit, _reserved, part) = (lock, permit, reserved, part);
            let mut file = BufWriter::with_capacity(256 * 1024, file);
            let mut upstream = object.body;
            let mut received = 0u64;
            while let Some(chunk) = upstream.try_next().await? {
                received += chunk.len() as u64;
                file.write_all(&chunk).await?;
                if received == length {
                    file.flush().await?;
                    file.get_ref().sync_all().await?;
                    fs::rename(&part.0, &path).await?;
                }
                yield chunk;
            }
            if received != length { Err(anyhow::anyhow!("truncated R2 asset range"))?; }
        }
        .map_err(|e: anyhow::Error| std::io::Error::other(e.to_string())),
    );
    response(source, range, body)
}

async fn cached(path: &Path) -> Result<Option<Body>> {
    match fs::File::open(path).await {
        Ok(file) => Ok(Some(Body::from_stream(
            tokio_util::io::ReaderStream::with_capacity(file, 256 * 1024),
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn response(source: AssetSource, range: (u64, u64), body: Body) -> Result<Response> {
    let (start, end) = range;
    let size = source.download.size;
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
        .and_then(|range| range.split_once('-'))
        .context("invalid asset range")?;
    let start: u64 = start.parse().context("invalid range start")?;
    let end: u64 = end.parse().context("invalid range end")?;
    ensure!(
        start <= end && end < size,
        "asset range is outside the object"
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
        assert_eq!(
            parse_range("bytes=0-4194304", 4194305).unwrap(),
            (0, 4194304)
        );
    }
}
