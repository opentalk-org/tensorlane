use anyhow::{Context, ensure};
use futures_util::{StreamExt, TryStreamExt, stream};
use reqwest::{Method, StatusCode};
use sha2::{Digest, Sha256};
use fs2::FileExt;
use std::{
    collections::HashMap,
    os::unix::fs::FileExt as _,
    path::{Path, PathBuf},
    sync::Arc,
};
use tensorlane_protocol::{AssetDownload, InitResponse, TRANSFER_CHUNK_BYTES};
use tokio::fs;

const PARALLEL_RANGES: usize = 16;

pub async fn prefetch(
    http: &crate::transport::HttpClient,
    initialized: &InitResponse,
    root: &Path,
) -> anyhow::Result<(HashMap<String, PathBuf>, HashMap<String, String>)> {
    let downloads: Vec<_> = stream::iter(initialized.assets.iter().enumerate())
        .map(|(index, name)| {
            let http = http.clone();
            let destination = root.join("assets").join(index.to_string());
            let input = initialized
                .checkpoint
                .as_ref()
                .filter(|checkpoint| checkpoint.name == *name)
                .map(|checkpoint| checkpoint.asset_id.as_str())
                .unwrap_or(name);
            async move {
                let (path, metadata) = download(http, &initialized.run_id, input, destination)
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
        .request(Method::GET, &["runs", run_id, "inputs", name], None, &[])
        .await?;
    let info: AssetDownload = serde_json::from_slice(&bytes)?;
    let entrypoint = info.metadata.entrypoint.clone();
    let asset_metadata: serde_json::Value = serde_json::from_str(&info.metadata.metadata_json)?;
    let metadata = serde_json::json!({
        "asset_id": info.metadata.asset_id,
        "entrypoint": info.metadata.entrypoint,
        "metadata": asset_metadata,
        "kind": info.metadata.kind,
        "asset_type": info.metadata.asset_type,
    })
    .to_string();

    // Asset bytes are immutable, so a finished entry is keyed by asset id and content hash.
    let cache = std::env::var_os("TENSORLANE_ASSET_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_else(|| "/root".into()))
                .join(".cache/tensorlane/assets")
        });
    fs::create_dir_all(&cache).await?;
    let cache = fs::canonicalize(cache).await?;
    let content = info.sha256.as_deref().unwrap_or(&info.etag);
    let asset_id = info.metadata.asset_id.as_deref().unwrap_or("");
    let key = hex::encode(Sha256::digest(format!("{asset_id}\n{content}")));
    let entry = cache.join(&key);

    if !entry.exists() {
        // The lock serializes processes; the entry only appears by atomic rename after a
        // complete download and extraction.
        let lock_path = cache.join(format!("{key}.lock"));
        let lock = tokio::task::spawn_blocking(move || {
            let lock = std::fs::OpenOptions::new()
                .create(true)
                .truncate(false)
                .write(true)
                .open(lock_path)?;
            lock.lock_exclusive()?;
            Ok::<_, std::io::Error>(lock)
        })
        .await??;
        if !entry.exists() {
            let partial = cache.join(format!("{key}.part"));
            let _ = fs::remove_dir_all(&partial).await;
            fs::create_dir_all(&partial).await?;
            let data = partial.join("data");
            fetch(&http, run_id, name, &info, &data).await?;
            let extract_to = partial.clone();
            tokio::task::spawn_blocking(move || extract(&extract_to))
                .await
                .context("asset extraction task failed")??;
            fs::rename(&partial, &entry).await?;
        }
        drop(lock);
    }

    let selected = select(&entry, entrypoint.as_deref())?;
    fs::create_dir_all(destination.parent().context("asset destination has no parent")?).await?;
    fs::symlink(&entry, &destination).await?;
    Ok((destination.join(selected.strip_prefix(&entry)?), metadata))
}

/// Download `info.size` bytes with parallel Range requests into a preallocated file.
async fn fetch(
    http: &crate::transport::HttpClient,
    run_id: &str,
    name: &str,
    info: &AssetDownload,
    data: &Path,
) -> anyhow::Result<()> {
    let file = std::fs::File::create(data)?;
    file.set_len(info.size)?;
    let file = Arc::new(file);
    let chunk = TRANSFER_CHUNK_BYTES as u64;
    stream::iter((0..info.size).step_by(TRANSFER_CHUNK_BYTES))
        .map(|offset| {
            let file = file.clone();
            async move {
                let end = (offset + chunk).min(info.size) - 1;
                let (status, headers, bytes) = http
                    .request(
                        Method::GET,
                        &["runs", run_id, "inputs", name, "bytes"],
                        None,
                        &[
                            ("range", format!("bytes={offset}-{end}")),
                            ("if-match", info.etag.clone()),
                        ],
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
                tokio::task::spawn_blocking(move || file.write_all_at(&bytes, offset))
                    .await
                    .context("asset write task failed")??;
                Ok::<_, anyhow::Error>(())
            }
        })
        .buffer_unordered(PARALLEL_RANGES)
        .try_collect::<()>()
        .await?;
    if let Some(expected) = info.sha256.clone() {
        let path = data.to_owned();
        let actual = tokio::task::spawn_blocking(move || {
            let mut digest = Sha256::new();
            std::io::copy(&mut std::fs::File::open(path)?, &mut digest)?;
            Ok::<_, std::io::Error>(hex::encode(digest.finalize()))
        })
        .await??;
        ensure!(actual == expected, "downloaded asset hash does not match metadata");
    }
    Ok(())
}

/// Unpack `<dir>/data` into `<dir>/extracted` and remove it when it is a tar archive.
fn extract(dir: &Path) -> anyhow::Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let path = dir.join("data");
    let mut file = std::fs::File::open(&path)?;
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
        return Ok(());
    }
    file.seek(SeekFrom::Start(0))?;
    let extracted = dir.join("extracted");
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
    std::fs::remove_file(path)?;
    Ok(())
}

/// The path handed to the training process: the raw file, or the entrypoint / single child
/// of the extracted archive.
fn select(dir: &Path, entrypoint: Option<&str>) -> anyhow::Result<PathBuf> {
    let data = dir.join("data");
    if data.exists() {
        return Ok(data);
    }
    let extracted = dir.join("extracted");
    if let Some(entrypoint) = entrypoint {
        let selected = extracted
            .join(entrypoint)
            .canonicalize()
            .context("asset entrypoint does not exist")?;
        anyhow::ensure!(
            selected.starts_with(extracted.canonicalize()?),
            "asset entrypoint escapes extraction directory"
        );
        return Ok(selected);
    }
    let children = std::fs::read_dir(&extracted)?.collect::<Result<Vec<_>, _>>()?;
    if children.len() == 1 {
        Ok(children[0].path())
    } else {
        Ok(extracted)
    }
}
