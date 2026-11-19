//! Resumable uploads (tus 1.0.0 with the creation and creation-with-upload
//! extensions) into the same final R2 objects as `PUT /uploads/{id}`.
//!
//! `POST /uploads` declares the upload (its ID and spec in `Upload-Metadata`) and may
//! carry its first bytes, so a small file is still one request. `PATCH
//! /uploads/{id}` appends at `Upload-Offset` and `HEAD /uploads/{id}` returns the
//! stored offset. Requests can reach any replica: the S3 multipart upload ID and,
//! per stored part, the SHA-256 state after that part are immutable R2 state
//! records, so the server still checks the whole file's SHA-256 before completing
//! the object. Bytes are stored in whole parts (`upload_part_bytes`); a request cut
//! inside a part stores nothing of that part. The last byte completes the object
//! and publishes its asset or artifact row; a `HEAD` or `PATCH` of a complete
//! upload publishes again, so a lost completion is finished by the next request.

use crate::{runtime::Runtime, upload_http};
use anyhow::{Context, Result, ensure};
use aws_sdk_s3::{
    primitives::ByteStream,
    types::{CompletedMultipartUpload, CompletedPart},
};
use axum::body::Body;
use bytes::{Bytes, BytesMut};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::Duration;
use tensorlane_protocol::{UploadSpec, upload_part_bytes};
use uuid::Uuid;

const SHA256_INITIAL: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

#[derive(Clone, Serialize, Deserialize)]
pub struct Session {
    pub spec: UploadSpec,
    fingerprint: String,
    upload_id: String,
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
struct StoredPart {
    sha256_state: [u32; 8],
}

fn session_key(id: Uuid) -> String {
    format!("tus/{id}")
}

fn part_key(id: Uuid, number: u64) -> String {
    format!("tus/{id}/parts/{number:05}")
}

/// Creates the upload, or returns it when a retry declares the same spec.
pub async fn create(engine: &Runtime, id: Uuid, mut spec: UploadSpec) -> Result<Session> {
    let (run, key, content_type) = upload_http::validate(engine, id, &mut spec)?;
    engine.active(run).await?;
    let fingerprint = upload_http::fingerprint(&spec)?;
    if let Some(session) = engine.read_state::<Session>(&session_key(id)).await? {
        ensure!(
            session.fingerprint == fingerprint,
            "conflicting retry of upload ID"
        );
        return Ok(session);
    }
    let upload_id = if spec.size == 0 {
        String::new()
    } else {
        engine
            .s3
            .create_multipart_upload()
            .bucket(engine.bucket)
            .key(&key)
            .content_type(content_type)
            .metadata("sha256", &spec.sha256)
            .metadata("spec", &fingerprint)
            .send()
            .await?
            .upload_id
            .context("missing S3 upload ID")?
    };
    let session = Session {
        spec,
        fingerprint,
        upload_id,
    };
    let stored = engine.create_state(&session_key(id), &session).await?;
    if stored.upload_id != session.upload_id && !session.upload_id.is_empty() {
        // Another replica created the upload first; use its multipart upload.
        let _ = engine
            .s3
            .abort_multipart_upload()
            .bucket(engine.bucket)
            .key(&key)
            .upload_id(&session.upload_id)
            .send()
            .await;
    }
    ensure!(
        stored.fingerprint == session.fingerprint,
        "conflicting retry of upload ID"
    );
    Ok(stored)
}

pub async fn session(engine: &Runtime, id: Uuid) -> Result<Session> {
    engine
        .read_state(&session_key(id))
        .await?
        .context("upload not found")
}

/// The upload's run, final R2 key and content type.
fn target(engine: &Runtime, id: Uuid, session: &Session) -> Result<(Uuid, String, String)> {
    upload_http::validate(engine, id, &mut session.spec.clone())
}

/// The stored bytes: the leading run of stored parts.
pub async fn offset(engine: &Runtime, id: Uuid, session: &Session) -> Result<u64> {
    let prefix = format!("tus/{id}/parts/");
    let mut numbers = Vec::new();
    for key in engine.list_state(&prefix).await? {
        if let Some(number) = key
            .strip_prefix(&prefix)
            .and_then(|n| n.parse::<u64>().ok())
        {
            numbers.push(number);
        }
    }
    numbers.sort_unstable();
    let stored = numbers
        .iter()
        .zip(1..)
        .take_while(|(number, expected)| **number == *expected)
        .count() as u64;
    Ok((stored * upload_part_bytes(session.spec.size)).min(session.spec.size))
}

/// Stores the body's bytes from `offset` (the stored offset) in whole parts, and
/// returns the new stored offset.
pub async fn receive(
    engine: &Runtime,
    id: Uuid,
    session: &Session,
    mut offset: u64,
    body: Body,
) -> Result<u64> {
    let (run, key, _) = target(engine, id, session)?;
    engine.active(run).await?;
    let size = session.spec.size;
    let part = upload_part_bytes(size);
    let mut number = offset / part + 1;
    let mut state = if number == 1 {
        SHA256_INITIAL
    } else {
        engine
            .read_state::<StoredPart>(&part_key(id, number - 1))
            .await?
            .context("upload part is missing")?
            .sha256_state
    };
    let mut buffer = BytesMut::new();
    let mut stream = body.into_data_stream();
    while let Some(chunk) = tokio::time::timeout(Duration::from_secs(30), stream.next()).await? {
        let chunk = chunk?;
        ensure!(
            offset + (buffer.len() + chunk.len()) as u64 <= size,
            "upload exceeds its length"
        );
        buffer.extend_from_slice(&chunk);
        while buffer.len() as u64 >= part && offset + part < size {
            let bytes = buffer.split_to(usize::try_from(part)?).freeze();
            compress(&mut state, &bytes);
            store(engine, id, session, &key, number, bytes, state).await?;
            offset += part;
            number += 1;
        }
    }
    if !buffer.is_empty() && offset + buffer.len() as u64 == size {
        let bytes = buffer.freeze();
        ensure!(
            hex::encode(finish_sha256(state, &bytes, size)) == session.spec.sha256,
            "upload SHA256 does not match"
        );
        store(engine, id, session, &key, number, bytes, state).await?;
        offset = size;
    }
    Ok(offset)
}

async fn store(
    engine: &Runtime,
    id: Uuid,
    session: &Session,
    key: &str,
    number: u64,
    bytes: Bytes,
    state: [u32; 8],
) -> Result<()> {
    engine
        .s3
        .upload_part()
        .bucket(engine.bucket)
        .key(key)
        .upload_id(&session.upload_id)
        .part_number(i32::try_from(number)?)
        .body(ByteStream::from(bytes))
        .send()
        .await?;
    let part = StoredPart {
        sha256_state: state,
    };
    let stored = engine.create_state(&part_key(id, number), &part).await?;
    ensure!(stored == part, "conflicting retry of upload ID");
    Ok(())
}

/// Completes the final object of a fully stored upload and publishes its row;
/// repeating it is harmless.
pub async fn finish(engine: &Runtime, id: Uuid, session: &Session) -> Result<()> {
    let (run, key, content_type) = target(engine, id, session)?;
    let spec = &session.spec;
    let s3 = &engine.s3;
    if !crate::s3_upload::existing(s3, engine.bucket, &key, spec, &session.fingerprint).await? {
        let completed = if spec.size == 0 {
            ensure!(
                spec.sha256 == hex::encode(Sha256::digest(b"")),
                "upload SHA256 does not match"
            );
            s3.put_object()
                .bucket(engine.bucket)
                .key(&key)
                .content_type(content_type)
                .metadata("sha256", &spec.sha256)
                .metadata("spec", &session.fingerprint)
                .if_none_match("*")
                .body(ByteStream::from_static(b""))
                .send()
                .await
                .map(|_| ())
                .map_err(anyhow::Error::from)
        } else {
            let mut parts = Vec::new();
            let mut marker = None;
            loop {
                let page = s3
                    .list_parts()
                    .bucket(engine.bucket)
                    .key(&key)
                    .upload_id(&session.upload_id)
                    .set_part_number_marker(marker)
                    .send()
                    .await?;
                for part in page.parts() {
                    parts.push(
                        CompletedPart::builder()
                            .set_part_number(part.part_number())
                            .set_e_tag(part.e_tag().map(str::to_owned))
                            .build(),
                    );
                }
                if !page.is_truncated().unwrap_or(false) {
                    break;
                }
                marker = page.next_part_number_marker().map(str::to_owned);
            }
            ensure!(
                parts.len() as u64 == spec.size.div_ceil(upload_part_bytes(spec.size)),
                "upload is incomplete"
            );
            s3.complete_multipart_upload()
                .bucket(engine.bucket)
                .key(&key)
                .upload_id(&session.upload_id)
                .multipart_upload(
                    CompletedMultipartUpload::builder()
                        .set_parts(Some(parts))
                        .build(),
                )
                .send()
                .await
                .map(|_| ())
                .map_err(anyhow::Error::from)
        };
        if let Err(error) = completed {
            ensure!(
                crate::s3_upload::existing(s3, engine.bucket, &key, spec, &session.fingerprint)
                    .await?,
                error
            );
        }
    }
    upload_http::commit(engine, id, spec, run, key).await
}

fn compress(state: &mut [u32; 8], bytes: &[u8]) {
    let blocks: Vec<_> = bytes
        .as_chunks::<64>()
        .0
        .iter()
        .map(|&block| block.into())
        .collect();
    sha2::compress256(state, &blocks);
}

/// The SHA-256 of a `size`-byte file from the state after all but its last `tail`
/// bytes (a whole number of 64-byte blocks).
fn finish_sha256(mut state: [u32; 8], tail: &[u8], size: u64) -> [u8; 32] {
    let whole = tail.len() / 64 * 64;
    compress(&mut state, &tail[..whole]);
    let mut last = tail[whole..].to_vec();
    last.push(0x80);
    while last.len() % 64 != 56 {
        last.push(0);
    }
    last.extend_from_slice(&(size * 8).to_be_bytes());
    compress(&mut state, &last);
    let mut digest = [0; 32];
    for (bytes, word) in digest.as_chunks_mut::<4>().0.iter_mut().zip(state) {
        *bytes = word.to_be_bytes();
    }
    digest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chained_sha256_matches_the_whole_file() {
        for (size, part) in [
            (0, 64),
            (1, 64),
            (55, 64),
            (56, 64),
            (64, 64),
            (200, 64),
            (1000, 128),
        ] {
            let bytes: Vec<u8> = (0..size).map(|i| (i * 7 + 3) as u8).collect();
            let mut state = SHA256_INITIAL;
            let mut offset = 0;
            while bytes.len() - offset > part {
                compress(&mut state, &bytes[offset..offset + part]);
                offset += part;
            }
            assert_eq!(
                hex::encode(finish_sha256(state, &bytes[offset..], size as u64)),
                hex::encode(Sha256::digest(&bytes)),
                "size {size}"
            );
        }
    }
}
