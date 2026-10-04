use crate::shared_cache::write_atomic;
use anyhow::{Context, Result, ensure};
use aws_sdk_s3::error::ProvideErrorMetadata;
use aws_sdk_s3::{
    Client,
    primitives::{ByteStream, Length},
    types::{CompletedMultipartUpload, CompletedPart},
};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::fs;

#[derive(Deserialize, Serialize)]
struct Multipart {
    upload_id: String,
    parts: Vec<String>,
}

pub async fn upload(
    client: &Client,
    bucket: &str,
    path: &Path,
    key: &str,
    content_type: &str,
) -> Result<()> {
    let size = fs::metadata(path).await?.len();
    let receipt = path.with_extension("s3-complete");
    if fs::try_exists(&receipt).await? {
        return Ok(());
    }
    if size == 0 {
        client
            .put_object()
            .bucket(bucket)
            .key(key)
            .content_type(content_type)
            .body(ByteStream::from_static(b""))
            .send()
            .await?;
        return write_atomic(&receipt, b"complete").await;
    }
    let progress = path.with_extension("multipart.json");
    let mut state: Multipart = if fs::try_exists(&progress).await? {
        serde_json::from_slice(&fs::read(&progress).await?)?
    } else {
        let response = client
            .create_multipart_upload()
            .bucket(bucket)
            .key(key)
            .content_type(content_type)
            .send()
            .await?;
        let state = Multipart {
            upload_id: response.upload_id().context("missing S3 upload ID")?.into(),
            parts: Vec::new(),
        };
        write_atomic(&progress, &serde_json::to_vec(&state)?).await?;
        state
    };
    let part_size = (16 * 1024 * 1024u64).max(size.div_ceil(10_000));
    ensure!(
        part_size <= 5 * 1024 * 1024 * 1024,
        "upload exceeds multipart limits"
    );
    for index in state.parts.len() as u64..size.div_ceil(part_size) {
        let offset = index * part_size;
        let body = ByteStream::read_from()
            .path(path)
            .offset(offset)
            .length(Length::Exact(part_size.min(size - offset)))
            .build()
            .await?;
        let response = client
            .upload_part()
            .bucket(bucket)
            .key(key)
            .upload_id(&state.upload_id)
            .part_number(i32::try_from(index + 1)?)
            .body(body)
            .send()
            .await;
        match response {
            Ok(part) => {
                state
                    .parts
                    .push(part.e_tag().context("missing S3 part ETag")?.into());
                write_atomic(&progress, &serde_json::to_vec(&state)?).await?;
            }
            Err(error)
                if error
                    .as_service_error()
                    .is_some_and(|error| error.code() == Some("NoSuchUpload")) =>
            {
                return completed(client, bucket, key, size, &progress, &receipt).await;
            }
            Err(error) => return Err(error.into()),
        }
    }
    let parts = state
        .parts
        .iter()
        .enumerate()
        .map(|(index, etag)| {
            CompletedPart::builder()
                .part_number((index + 1) as i32)
                .e_tag(etag)
                .build()
        })
        .collect();
    let response = client
        .complete_multipart_upload()
        .bucket(bucket)
        .key(key)
        .upload_id(&state.upload_id)
        .multipart_upload(
            CompletedMultipartUpload::builder()
                .set_parts(Some(parts))
                .build(),
        )
        .send()
        .await;
    match response {
        Ok(_) => write_atomic(&receipt, b"complete").await,
        Err(error)
            if error
                .as_service_error()
                .is_some_and(|error| error.code() == Some("NoSuchUpload")) =>
        {
            completed(client, bucket, key, size, &progress, &receipt).await
        }
        Err(error) => Err(error.into()),
    }
}

async fn completed(
    client: &Client,
    bucket: &str,
    key: &str,
    size: u64,
    progress: &Path,
    receipt: &Path,
) -> Result<()> {
    let object = client.head_object().bucket(bucket).key(key).send().await;
    if object
        .as_ref()
        .is_ok_and(|head| head.content_length() == Some(size as i64))
    {
        return write_atomic(receipt, b"complete").await;
    }
    fs::remove_file(progress).await?;
    anyhow::bail!("multipart upload expired; restarting from retained chunks")
}
