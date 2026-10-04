use anyhow::{Context, ensure};
use futures_util::{StreamExt, TryStreamExt, stream};
use reqwest::{Method, StatusCode};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
use tensorlane_protocol::{AssetDownload, InitResponse, TRANSFER_CHUNK_BYTES};
use tokio::{fs, io::AsyncWriteExt};

pub async fn prefetch(
    http: &crate::transport::HttpClient,
    initialized: &InitResponse,
    root: &Path,
) -> anyhow::Result<(HashMap<String, PathBuf>, HashMap<String, String>)> {
    let downloads: Vec<_> = stream::iter(initialized.assets.iter().enumerate())
        .map(|(index, name)| {
            let http = http.clone();
            let destination = root.join("assets").join(index.to_string());
            async move {
                let (path, metadata) = download(http, &initialized.run_id, name, destination)
                    .await
                    .with_context(|| format!("downloading asset {name:?}"))?;
                Ok::<_, anyhow::Error>((name.clone(), path, metadata))
            }
        })
        .buffer_unordered(4)
        .try_collect()
        .await?;
    let mut paths = HashMap::new();
    let mut metadata = HashMap::new();
    for (name, path, info) in downloads {
        paths.insert(name.clone(), path);
        metadata.insert(name, info);
    }
    Ok((paths, metadata))
}
async fn download(
    http: crate::transport::HttpClient,
    run_id: &str,
    name: &str,
    destination: PathBuf,
) -> anyhow::Result<(PathBuf, String)> {
    let (_, _, bytes) = http
        .request(
            Method::GET,
            &["runs", run_id, "inputs", name],
            None,
            &[],
            1024 * 1024,
        )
        .await?;
    let info: AssetDownload = serde_json::from_slice(&bytes)?;
    let entrypoint = info.metadata.entrypoint.clone();
    let metadata =
        serde_json::json!({"asset_id":info.metadata.asset_id,"entrypoint":info.metadata.entrypoint,
        "metadata":serde_json::from_str::<serde_json::Value>(&info.metadata.metadata_json)?,
        "kind":info.metadata.kind,"asset_type":info.metadata.asset_type})
        .to_string();
    fs::create_dir_all(&destination).await?;
    let partial = destination.join("download.part");
    let path = destination.join("data");
    let mut file = fs::File::create(&partial).await?;
    let mut offset = 0u64;
    let mut digest = Sha256::new();
    while offset < info.size {
        let end = (offset + TRANSFER_CHUNK_BYTES as u64).min(info.size) - 1;
        let (status, headers, bytes) = http
            .request(
                Method::GET,
                &["runs", run_id, "inputs", name, "bytes"],
                None,
                &[
                    ("range", format!("bytes={offset}-{end}")),
                    ("if-match", info.etag.clone()),
                ],
                TRANSFER_CHUNK_BYTES,
            )
            .await?;
        ensure!(
            status == StatusCode::PARTIAL_CONTENT,
            "server did not honor asset Range"
        );
        ensure!(
            headers.get("content-range").and_then(|v| v.to_str().ok())
                == Some(format!("bytes {offset}-{end}/{}", info.size).as_str()),
            "unexpected asset Content-Range"
        );
        ensure!(
            headers.get("etag").and_then(|v| v.to_str().ok()) == Some(info.etag.as_str()),
            "input asset changed during download"
        );
        ensure!(
            bytes.len() as u64 == end - offset + 1,
            "truncated asset range"
        );
        file.write_all(&bytes).await?;
        digest.update(&bytes);
        offset = end + 1;
    }
    if let Some(expected) = info.sha256 {
        ensure!(
            hex::encode(digest.finalize()) == expected,
            "downloaded asset hash does not match metadata"
        );
    }
    file.flush().await?;
    drop(file);
    fs::rename(partial, &path).await?;
    let path = tokio::task::spawn_blocking(move || materialize(&path, entrypoint.as_deref()))
        .await
        .context("asset extraction task failed")??;
    Ok((fs::canonicalize(path).await?, metadata))
}

fn materialize(path: &Path, entrypoint: Option<&str>) -> anyhow::Result<PathBuf> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path)?;
    let mut block = [0u8; 512];
    let count = file.read(&mut block)?;
    let header = tar::Header::from_byte_slice(&block);
    let checksum = block[..148]
        .iter()
        .chain(&block[156..])
        .map(|byte| u32::from(*byte))
        .sum::<u32>()
        + 8 * 32;
    let is_tar = count == 512
        && (header.cksum().ok() == Some(checksum)
            || (block == [0; 512] && file.metadata()?.len() >= 1024));
    if !is_tar {
        return Ok(path.to_owned());
    }
    file.seek(SeekFrom::Start(0))?;
    let extracted = path.with_file_name("extracted");
    std::fs::create_dir_all(&extracted)?;
    for entry in tar::Archive::new(file).entries()? {
        let mut entry = entry?;
        anyhow::ensure!(
            entry.header().entry_type().is_file() || entry.header().entry_type().is_dir(),
            "asset archive entries must be regular files or directories"
        );
        anyhow::ensure!(
            entry.unpack_in(&extracted)?,
            "asset archive path escapes extraction directory"
        );
    }
    let result = if let Some(entrypoint) = entrypoint {
        let selected = extracted
            .join(entrypoint)
            .canonicalize()
            .context("asset entrypoint does not exist")?;
        anyhow::ensure!(
            selected.starts_with(extracted.canonicalize()?),
            "asset entrypoint escapes extraction directory"
        );
        selected
    } else {
        let children = std::fs::read_dir(&extracted)?.collect::<Result<Vec<_>, _>>()?;
        if children.len() == 1 {
            children[0].path()
        } else {
            extracted
        }
    };
    std::fs::remove_file(path)?;
    Ok(result)
}
