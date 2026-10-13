use anyhow::{Context, Result, ensure};
use aws_sdk_s3::{
    Client,
    primitives::ByteStream,
    types::{CompletedMultipartUpload, CompletedPart},
};
use axum::extract::Multipart;
use bytes::{Bytes, BytesMut};
use sha2::{Digest, Sha256};
use std::time::Duration;
use tensorlane_protocol::UploadSpec;

const PART_BYTES: usize = 16 * 1024 * 1024;

pub async fn upload(
    client: &Client,
    bucket: &str,
    key: &str,
    content_type: &str,
    spec: &UploadSpec,
    fingerprint: &str,
    multipart: &mut Multipart,
) -> Result<()> {
    if existing(client, bucket, key, spec, fingerprint).await? {
        receive(client, bucket, key, spec, multipart, None).await?;
        return Ok(());
    }
    if spec.size == 0 {
        receive(client, bucket, key, spec, multipart, None).await?;
        let response = client
            .put_object()
            .bucket(bucket)
            .key(key)
            .content_type(content_type)
            .metadata("sha256", &spec.sha256)
            .metadata("spec", fingerprint)
            .if_none_match("*")
            .body(ByteStream::from_static(b""))
            .send()
            .await;
        if let Err(error) = response
            && !existing(client, bucket, key, spec, fingerprint).await?
        {
            return Err(error.into());
        }
        return Ok(());
    }
    let upload_id = client
        .create_multipart_upload()
        .bucket(bucket)
        .key(key)
        .content_type(content_type)
        .metadata("sha256", &spec.sha256)
        .metadata("spec", fingerprint)
        .send()
        .await?
        .upload_id
        .context("missing S3 upload ID")?;
    let abort = client
        .abort_multipart_upload()
        .bucket(bucket)
        .key(key)
        .upload_id(&upload_id);
    let parts = match receive(client, bucket, key, spec, multipart, Some(&upload_id)).await {
        Ok(parts) => parts,
        Err(error) => {
            let _ = abort.send().await;
            return Err(error);
        }
    };
    let completed = client
        .complete_multipart_upload()
        .bucket(bucket)
        .key(key)
        .upload_id(&upload_id)
        .multipart_upload(
            CompletedMultipartUpload::builder()
                .set_parts(Some(parts))
                .build(),
        )
        .send()
        .await;
    if let Err(error) = completed {
        let _ = abort.send().await;
        ensure!(
            existing(client, bucket, key, spec, fingerprint).await?,
            error
        );
    }
    Ok(())
}

async fn receive(
    client: &Client,
    bucket: &str,
    key: &str,
    spec: &UploadSpec,
    multipart: &mut Multipart,
    upload_id: Option<&str>,
) -> Result<Vec<CompletedPart>> {
    let part_bytes = PART_BYTES.max(usize::try_from(spec.size.div_ceil(10_000))?);
    let mut field = multipart
        .next_field()
        .await?
        .context("upload is missing its file")?;
    ensure!(field.name() == Some("file"), "upload is missing its file");
    let mut digest = Sha256::new();
    let mut size = 0;
    let mut parts = Vec::new();
    let mut buffer = BytesMut::new();
    while let Some(chunk) = tokio::time::timeout(Duration::from_secs(30), field.chunk()).await?? {
        size += chunk.len() as u64;
        ensure!(size <= spec.size, "upload has unexpected size");
        digest.update(&chunk);
        let Some(upload_id) = upload_id else {
            continue;
        };
        buffer.extend_from_slice(&chunk);
        while buffer.len() >= part_bytes {
            let bytes = buffer.split_to(part_bytes).freeze();
            parts.push(part(client, bucket, key, upload_id, parts.len() + 1, bytes).await?);
        }
    }
    drop(field);
    ensure!(size == spec.size, "upload has unexpected size");
    ensure!(
        hex::encode(digest.finalize()) == spec.sha256,
        "upload SHA256 does not match"
    );
    ensure!(
        multipart.next_field().await?.is_none(),
        "upload contains unexpected fields"
    );
    if let Some(upload_id) = upload_id
        && !buffer.is_empty()
    {
        parts.push(
            part(
                client,
                bucket,
                key,
                upload_id,
                parts.len() + 1,
                buffer.freeze(),
            )
            .await?,
        );
    }
    Ok(parts)
}

async fn part(
    client: &Client,
    bucket: &str,
    key: &str,
    upload_id: &str,
    number: usize,
    bytes: Bytes,
) -> Result<CompletedPart> {
    let number = i32::try_from(number)?;
    let response = client
        .upload_part()
        .bucket(bucket)
        .key(key)
        .upload_id(upload_id)
        .part_number(number)
        .body(ByteStream::from(bytes))
        .send()
        .await?;
    Ok(CompletedPart::builder()
        .part_number(number)
        .e_tag(response.e_tag.context("missing S3 part ETag")?)
        .build())
}

async fn existing(
    client: &Client,
    bucket: &str,
    key: &str,
    spec: &UploadSpec,
    fingerprint: &str,
) -> Result<bool> {
    match client.head_object().bucket(bucket).key(key).send().await {
        Ok(head) => {
            let metadata = head.metadata.unwrap_or_default();
            ensure!(
                head.content_length == Some(spec.size as i64)
                    && metadata.get("sha256") == Some(&spec.sha256)
                    && metadata.get("spec").map(String::as_str) == Some(fingerprint),
                "conflicting retry of upload ID"
            );
            Ok(true)
        }
        Err(error)
            if error
                .raw_response()
                .is_some_and(|r| r.status().as_u16() == 404) =>
        {
            Ok(false)
        }
        Err(error) => Err(error.into()),
    }
}
