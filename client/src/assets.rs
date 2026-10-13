use anyhow::{Context, ensure};
use futures_util::{StreamExt, TryStreamExt, stream};
use reqwest::{Method, StatusCode};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    io::{self, Read, Seek, Write},
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    sync::{Arc, mpsc},
    time::SystemTime,
};
use tensorlane_protocol::{AssetDownload, InitResponse, TRANSFER_CHUNK_BYTES};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

const DEFAULT_CACHE_BYTES: u64 = 40 * 1024 * 1024 * 1024;

pub struct Prefetched {
    pub paths: HashMap<String, PathBuf>,
    pub metadata: HashMap<String, String>,
    pub leases: Vec<Lease>,
}

pub struct Lease {
    _lock: File,
    entry: PathBuf,
}

impl Drop for Lease {
    fn drop(&mut self) {
        let _ = File::open(&self.entry).and_then(|entry| entry.set_modified(SystemTime::now()));
    }
}

struct Cache {
    dir: PathBuf,
    limit: u64,
    memory: Arc<Semaphore>,
    chunk: usize,
}

fn cache_dir() -> io::Result<PathBuf> {
    let home = std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache"));
    let dir = std::env::var_os("TENSORLANE_ASSET_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            home.unwrap_or_else(std::env::temp_dir)
                .join("tensorlane/assets")
        });
    std::fs::create_dir_all(&dir)?;
    dir.canonicalize()
}

fn cache_limit() -> anyhow::Result<u64> {
    std::env::var("TENSORLANE_ASSET_CACHE_MAX_BYTES")
        .map(|limit| limit.parse())
        .unwrap_or(Ok(DEFAULT_CACHE_BYTES))
        .context("invalid TENSORLANE_ASSET_CACHE_MAX_BYTES")
}

fn cache_key(asset_id: &str, content: &str) -> String {
    hex::encode(Sha256::digest(format!("{asset_id}\n{content}")))
}

pub fn store(asset_id: &str, sha256: &str, mut source: File) -> anyhow::Result<()> {
    let dir = cache_dir()?;
    let key = cache_key(asset_id, sha256);
    let entry = dir.join(&key);
    let _lock = lock(&dir.join(format!("{key}.lock")))?;
    if entry.exists() {
        return Ok(());
    }
    let partial = entry.with_extension("part");
    let size = source.metadata()?.len();
    evict(&dir, cache_limit()?, size)?;
    let _ = std::fs::remove_dir_all(&partial);
    std::fs::create_dir_all(&partial)?;
    source.seek(io::SeekFrom::Start(0))?;
    unpack(&partial, &mut source, size)?;
    std::fs::rename(&partial, &entry)?;
    Ok(())
}

pub async fn prefetch(
    http: &crate::transport::HttpClient,
    initialized: &InitResponse,
    memory_bytes: usize,
) -> anyhow::Result<Prefetched> {
    let cache = Cache {
        dir: blocking(cache_dir).await.context("creating asset cache")?,
        limit: cache_limit()?,
        memory: Arc::new(Semaphore::new(
            memory_bytes.clamp(1, Semaphore::MAX_PERMITS),
        )),
        chunk: TRANSFER_CHUNK_BYTES.min(memory_bytes).max(1),
    };
    let downloads: Vec<_> = stream::iter(initialized.assets.iter())
        .map(|name| {
            let input = initialized
                .checkpoint
                .as_ref()
                .filter(|checkpoint| checkpoint.name == *name)
                .map(|checkpoint| checkpoint.asset_id.as_str())
                .unwrap_or(name);
            let cache = &cache;
            async move {
                let (path, metadata, lease) = download(http, cache, &initialized.run_id, input)
                    .await
                    .with_context(|| format!("downloading asset {name:?}"))?;
                Ok::<_, anyhow::Error>((name.clone(), path, metadata, lease))
            }
        })
        .buffer_unordered(8)
        .try_collect()
        .await?;
    let mut prefetched = Prefetched {
        paths: HashMap::new(),
        metadata: HashMap::new(),
        leases: Vec::new(),
    };
    for (name, path, info, lease) in downloads {
        prefetched.paths.insert(name.clone(), path);
        prefetched.metadata.insert(name, info);
        prefetched.leases.push(lease);
    }
    Ok(prefetched)
}

async fn download(
    http: &crate::transport::HttpClient,
    cache: &Cache,
    run_id: &str,
    name: &str,
) -> anyhow::Result<(PathBuf, String, Lease)> {
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

    let key = cache_key(
        info.metadata.asset_id.as_deref().unwrap_or(""),
        info.sha256.as_deref().unwrap_or(&info.etag),
    );
    let entry = cache.dir.join(&key);
    let lock_path = cache.dir.join(format!("{key}.lock"));
    let lock = loop {
        let path = lock_path.clone();
        let lock = blocking(move || lock(&path)).await?;
        if !entry.exists() {
            fill(http, cache, run_id, name, &info, &entry)
                .await
                .inspect_err(|_| {
                    let _ = std::fs::remove_dir_all(entry.with_extension("part"));
                    let _ = std::fs::remove_file(&lock_path);
                })?;
        }
        let used = entry.clone();
        let shared = blocking(move || {
            lock.lock_shared()?;
            let touched = File::open(&used).and_then(|entry| entry.set_modified(SystemTime::now()));
            Ok(touched.is_ok().then_some(lock))
        })
        .await?;
        if let Some(lock) = shared {
            break lock;
        }
    };
    let used = entry.clone();
    let selected = tokio::task::spawn_blocking(move || select(&used, entrypoint.as_deref()))
        .await
        .context("asset selection task failed")??;
    Ok((selected, metadata, Lease { _lock: lock, entry }))
}

async fn fill(
    http: &crate::transport::HttpClient,
    cache: &Cache,
    run_id: &str,
    name: &str,
    info: &AssetDownload,
    entry: &Path,
) -> anyhow::Result<()> {
    let partial = entry.with_extension("part");
    let (dir, limit, size, part) = (cache.dir.clone(), cache.limit, info.size, partial.clone());
    blocking(move || {
        evict(&dir, limit, size)?;
        let _ = std::fs::remove_dir_all(&part);
        std::fs::create_dir_all(&part)
    })
    .await?;
    let (sender, chunks) = mpsc::channel();
    let target = partial.clone();
    let unpacking = tokio::task::spawn_blocking(move || {
        let mut ordered = Ordered {
            chunks,
            ready: BTreeMap::new(),
            current: (io::Cursor::new(Vec::new()), None),
            next: 0,
            digest: Sha256::new(),
        };
        unpack(&target, &mut ordered, size)?;
        ensure!(ordered.next == size, "asset download ended early");
        anyhow::Ok(hex::encode(ordered.digest.finalize()))
    });
    let fetched = fetch(http, cache, run_id, name, info, sender).await;
    let unpacked = unpacking.await.context("asset unpack task failed")?;
    fetched?;
    let actual = unpacked?;
    ensure!(
        info.sha256
            .as_ref()
            .is_none_or(|expected| *expected == actual),
        "downloaded asset hash does not match metadata"
    );
    tokio::fs::rename(&partial, entry).await?;
    Ok(())
}

async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> io::Result<T> + Send + 'static,
) -> anyhow::Result<T> {
    Ok(tokio::task::spawn_blocking(work)
        .await
        .context("asset cache task failed")??)
}

fn lock(path: &Path) -> io::Result<File> {
    loop {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path)?;
        file.lock()?;
        let (held, current) = (file.metadata()?, std::fs::metadata(path));
        if current.is_ok_and(|current| current.ino() == held.ino() && current.dev() == held.dev()) {
            return Ok(file);
        }
    }
}

fn evict(dir: &Path, limit: u64, incoming: u64) -> io::Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(Result::ok)
        .map(|item| item.path())
        .filter(|path| path.is_dir())
        .filter_map(|path| {
            let used = std::fs::metadata(&path).and_then(|m| m.modified()).ok()?;
            Some((used, disk_usage(&path), path))
        })
        .collect();
    entries.sort();
    let mut total: u64 = entries.iter().map(|(_, size, _)| size).sum();
    let mut free = fs2::available_space(dir)?;
    let reserve = fs2::total_space(dir)? / 20;
    for (_, size, path) in entries {
        if total.saturating_add(incoming) <= limit && free >= incoming.saturating_add(reserve) {
            break;
        }
        let lock_path = path.with_extension("lock");
        let Ok(lock) = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)
        else {
            continue;
        };
        let Ok(()) = lock.try_lock() else {
            continue;
        };
        std::fs::remove_dir_all(&path)?;
        let _ = std::fs::remove_file(&lock_path);
        total -= size;
        free += size;
    }
    if free < incoming {
        return Err(io::Error::new(
            io::ErrorKind::StorageFull,
            format!(
                "asset needs {incoming} bytes but {} has {free} free after evicting unused assets",
                dir.display()
            ),
        ));
    }
    Ok(())
}

fn disk_usage(path: &Path) -> u64 {
    let Ok(children) = std::fs::read_dir(path) else {
        return std::fs::metadata(path).map_or(0, |metadata| metadata.len());
    };
    children
        .filter_map(Result::ok)
        .map(|child| disk_usage(&child.path()))
        .sum()
}

async fn fetch(
    http: &crate::transport::HttpClient,
    cache: &Cache,
    run_id: &str,
    name: &str,
    info: &AssetDownload,
    chunks: mpsc::Sender<(u64, Vec<u8>, OwnedSemaphorePermit)>,
) -> anyhow::Result<()> {
    let (size, chunk) = (info.size, cache.chunk as u64);
    stream::iter((0..size).step_by(cache.chunk))
        .map(|offset| {
            let chunks = chunks.clone();
            async move {
                let end = (offset + chunk).min(size) - 1;
                let reserved = cache
                    .memory
                    .clone()
                    .acquire_many_owned((end - offset + 1) as u32)
                    .await?;
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
                        == Some(format!("bytes {offset}-{end}/{size}").as_str()),
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
                let _ = chunks.send((offset, bytes, reserved));
                Ok::<_, anyhow::Error>(())
            }
        })
        .buffer_unordered(1024)
        .try_collect()
        .await
}

struct Ordered {
    chunks: mpsc::Receiver<(u64, Vec<u8>, OwnedSemaphorePermit)>,
    ready: BTreeMap<u64, (Vec<u8>, OwnedSemaphorePermit)>,
    current: (io::Cursor<Vec<u8>>, Option<OwnedSemaphorePermit>),
    next: u64,
    digest: Sha256,
}

impl Read for Ordered {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        loop {
            let count = self.current.0.read(buf)?;
            if count > 0 {
                self.digest.update(&buf[..count]);
                return Ok(count);
            }
            if let Some((bytes, reserved)) = self.ready.remove(&self.next) {
                self.next += bytes.len() as u64;
                self.current = (io::Cursor::new(bytes), Some(reserved));
                continue;
            }
            let Ok((offset, bytes, reserved)) = self.chunks.recv() else {
                return Ok(0);
            };
            self.ready.insert(offset, (bytes, reserved));
        }
    }
}

fn unpack(dir: &Path, source: &mut impl Read, size: u64) -> anyhow::Result<()> {
    let mut head = Vec::new();
    source.by_ref().take(512).read_to_end(&mut head)?;
    let mut block = [0u8; 512];
    block[..head.len()].copy_from_slice(&head);
    let checksum = block[..148].iter().chain(&block[156..]);
    let checksum = checksum.map(|byte| u32::from(*byte)).sum::<u32>() + 8 * 32;
    let header = tar::Header::from_byte_slice(&block);
    let is_tar = head.len() == 512
        && (header.cksum().ok() == Some(checksum) || (block == [0; 512] && size >= 1024));
    let mut source = io::Cursor::new(head).chain(source);
    if !is_tar {
        let mut data = io::BufWriter::with_capacity(1 << 20, File::create(dir.join("data"))?);
        io::copy(&mut source, &mut data)?;
        data.flush()?;
        return Ok(());
    }
    let extracted = dir.join("extracted");
    std::fs::create_dir_all(&extracted)?;
    for entry in tar::Archive::new(&mut source).entries()? {
        let mut entry = entry?;
        ensure!(
            entry.header().entry_type().is_file() || entry.header().entry_type().is_dir(),
            "asset archive entries must be regular files or directories"
        );
        ensure!(
            entry.unpack_in(&extracted)?,
            "asset archive path escapes extraction directory"
        );
    }
    io::copy(&mut source, &mut io::sink())?;
    Ok(())
}

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
