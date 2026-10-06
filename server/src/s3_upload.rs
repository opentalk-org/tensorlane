use anyhow::{Context, Result, ensure};
use aws_sdk_s3::{
    Client,
    primitives::ByteStream,
    types::{CompletedMultipartUpload, CompletedPart},
};
use axum::extract::Multipart;
use bytes::Bytes;
use sha2::{Digest, Sha256};
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
    let part_bytes = PART_BYTES.max(usize::try_from(spec.size.div_ceil(10_000))?);
    let exists = existing(client, bucket, key, spec, fingerprint).await?;
    let upload_id = if !exists && spec.size > 0 {
        Some(
            client
                .create_multipart_upload()
                .bucket(bucket)
                .key(key)
                .content_type(content_type)
                .metadata("sha256", &spec.sha256)
                .metadata("spec", fingerprint)
                .send()
                .await?
                .upload_id()
                .context("missing S3 upload ID")?
                .to_owned(),
        )
    } else {
        None
    };
    let result = async {
        let mut field = multipart
            .next_field()
            .await?
            .context("upload is missing its file")?;
        ensure!(field.name() == Some("file"), "upload is missing its file");
        let mut digest = Sha256::new();
        let mut size = 0u64;
        let mut parts = Vec::new();
        let mut buffer = if upload_id.is_some() {
            Vec::with_capacity(part_bytes)
        } else {
            Vec::new()
        };
        while let Some(chunk) =
            tokio::time::timeout(std::time::Duration::from_secs(30), field.chunk()).await??
        {
            size = size
                .checked_add(chunk.len() as u64)
                .context("upload has unexpected size")?;
            ensure!(size <= spec.size, "upload has unexpected size");
            digest.update(&chunk);
            if let Some(upload_id) = &upload_id {
                let mut remaining = chunk.as_ref();
                while !remaining.is_empty() {
                    let count = remaining.len().min(part_bytes - buffer.len());
                    buffer.extend_from_slice(&remaining[..count]);
                    remaining = &remaining[count..];
                    if buffer.len() == part_bytes {
                        parts.push(
                            part(
                                client,
                                bucket,
                                key,
                                upload_id,
                                parts.len() + 1,
                                Bytes::from(std::mem::take(&mut buffer)),
                            )
                            .await?,
                        );
                        buffer = Vec::with_capacity(part_bytes);
                    }
                }
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
        if exists {
            return Ok(());
        }
        if let Some(upload_id) = &upload_id {
            if !buffer.is_empty() {
                parts.push(
                    part(
                        client,
                        bucket,
                        key,
                        upload_id,
                        parts.len() + 1,
                        Bytes::from(buffer),
                    )
                    .await?,
                );
            }
            let response = client
                .complete_multipart_upload()
                .bucket(bucket)
                .key(key)
                .upload_id(upload_id)
                .multipart_upload(
                    CompletedMultipartUpload::builder()
                        .set_parts(Some(parts))
                        .build(),
                )
                .send()
                .await;
            if let Err(error) = response {
                if !existing(client, bucket, key, spec, fingerprint).await? {
                    return Err(error.into());
                }
                let _ = client
                    .abort_multipart_upload()
                    .bucket(bucket)
                    .key(key)
                    .upload_id(upload_id)
                    .send()
                    .await;
            }
        } else {
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
        }
        Ok(())
    }
    .await;
    if result.is_err()
        && let Some(upload_id) = &upload_id
    {
        let _ = client
            .abort_multipart_upload()
            .bucket(bucket)
            .key(key)
            .upload_id(upload_id)
            .send()
            .await;
    }
    result
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
        .e_tag(response.e_tag().context("missing S3 part ETag")?)
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
            ensure!(
                head.content_length() == Some(spec.size as i64)
                    && head
                        .metadata()
                        .and_then(|m| m.get("sha256"))
                        .map(String::as_str)
                        == Some(spec.sha256.as_str())
                    && head
                        .metadata()
                        .and_then(|m| m.get("spec"))
                        .map(String::as_str)
                        == Some(fingerprint),
                "conflicting retry of upload ID"
            );
            Ok(true)
        }
        Err(error)
            if error
                .raw_response()
                .is_some_and(|response| response.status().as_u16() == 404) =>
        {
            Ok(false)
        }
        Err(error) => Err(error.into()),
    }
}
