use crate::transport::HttpClient;
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
use tokio::{
    sync::{OwnedSemaphorePermit, Semaphore},
    task::spawn_blocking,
};

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
    chunk: u64,
}

type Chunk = (u64, Vec<u8>, OwnedSemaphorePermit);

fn cache() -> anyhow::Result<(PathBuf, u64)> {
    let dir = match std::env::var_os("TENSORLANE_ASSET_CACHE") {
        Some(dir) => PathBuf::from(dir),
        None => std::env::var_os("HOME")
            .map_or_else(std::env::temp_dir, |home| {
                PathBuf::from(home).join(".cache")
            })
            .join("tensorlane/assets"),
    };
    std::fs::create_dir_all(&dir).context("creating asset cache")?;
    let limit = match std::env::var("TENSORLANE_ASSET_CACHE_MAX_BYTES") {
        Ok(limit) => limit
            .parse()
            .context("invalid TENSORLANE_ASSET_CACHE_MAX_BYTES")?,
        Err(_) => 40 << 30,
    };
    Ok((dir.canonicalize()?, limit))
}

fn cache_key(asset_id: &str, content: &str) -> String {
    hex::encode(Sha256::digest(format!("{asset_id}\n{content}")))
}

pub fn store(asset_id: &str, sha256: &str, mut source: File) -> anyhow::Result<()> {
    let (dir, limit) = cache()?;
    let key = cache_key(asset_id, sha256);
    let entry = dir.join(&key);
    let _lock = lock(&dir.join(format!("{key}.lock")))?;
    if entry.exists() {
        return Ok(());
    }
    let partial = entry.with_extension("part");
    let size = source.metadata()?.len();
    evict(&dir, limit, size)?;
    let _ = std::fs::remove_dir_all(&partial);
    std::fs::create_dir_all(&partial)?;
    source.rewind()?;
    unpack(&partial, &mut source, size)?;
    std::fs::rename(&partial, &entry)?;
    Ok(())
}

pub async fn prefetch(
    http: &HttpClient,
    initialized: &InitResponse,
    memory_bytes: usize,
) -> anyhow::Result<Prefetched> {
    let (dir, limit) = spawn_blocking(cache).await??;
    let cache = Cache {
        dir,
        limit,
        memory: Arc::new(Semaphore::new(
            memory_bytes.clamp(1, Semaphore::MAX_PERMITS),
        )),
        chunk: TRANSFER_CHUNK_BYTES.min(memory_bytes).max(1) as u64,
    };
    let downloads: Vec<_> = stream::iter(&initialized.assets)
        .map(|name| download(http, &cache, initialized, name))
        .buffer_unordered(8)
        .try_collect()
        .await?;
    let mut prefetched = Prefetched {
        paths: HashMap::new(),
        metadata: HashMap::new(),
        leases: Vec::new(),
    };
    for (name, path, metadata, lease) in downloads {
        prefetched.paths.insert(name.clone(), path);
        prefetched.metadata.insert(name, metadata);
        prefetched.leases.push(lease);
    }
    Ok(prefetched)
}

async fn download(
    http: &HttpClient,
    cache: &Cache,
    initialized: &InitResponse,
    name: &str,
) -> anyhow::Result<(String, PathBuf, String, Lease)> {
    let input = match &initialized.checkpoint {
        Some(checkpoint) if checkpoint.name == name => &checkpoint.asset_id,
        _ => name,
    };
    let run_id = &initialized.run_id;
    let context = format!("downloading asset {name:?}");
    let (_, _, bytes) = http
        .request(Method::GET, &["runs", run_id, "inputs", input], None, &[])
        .await
        .context(context.clone())?;
    let info: AssetDownload = serde_json::from_slice(&bytes)?;
    let metadata = serde_json::json!({
        "asset_id": info.metadata.asset_id,
        "entrypoint": info.metadata.entrypoint,
        "metadata": serde_json::from_str::<serde_json::Value>(&info.metadata.metadata_json)?,
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
        let mut held = spawn_blocking(move || lock(&path)).await??;
        let waited = entry.clone();
        held = spawn_blocking(move || {
            held.unlock()?;
            while !waited.exists() && held.try_lock().is_err() {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            io::Result::Ok(held)
        })
        .await??;
        if !entry.exists() {
            fill(http, cache, run_id, input, &info, &entry)
                .await
                .context(context.clone())?;
        }
        let used = entry.clone();
        let (shared, touched) = spawn_blocking(move || {
            held.lock_shared()?;
            let touched = File::open(&used).and_then(|used| used.set_modified(SystemTime::now()));
            io::Result::Ok((held, touched.is_ok()))
        })
        .await??;
        if touched {
            break shared;
        }
    };
    let (used, entrypoint) = (entry.clone(), info.metadata.entrypoint);
    let selected = spawn_blocking(move || select(&used, entrypoint.as_deref())).await??;
    Ok((
        name.to_owned(),
        selected,
        metadata,
        Lease { _lock: lock, entry },
    ))
}

async fn fill(
    http: &HttpClient,
    cache: &Cache,
    run_id: &str,
    input: &str,
    info: &AssetDownload,
    entry: &Path,
) -> anyhow::Result<()> {
    let partial = entry.with_extension("part");
    let (dir, limit, size, target) = (cache.dir.clone(), cache.limit, info.size, partial.clone());
    spawn_blocking(move || {
        evict(&dir, limit, size)?;
        let _ = std::fs::remove_dir_all(&target);
        std::fs::create_dir_all(&target)
    })
    .await??;
    let (sender, chunks) = mpsc::channel();
    let target = partial.clone();
    let unpacking = spawn_blocking(move || {
        let mut ordered = Ordered {
            chunks,
            ready: BTreeMap::new(),
            current: (io::Cursor::new(Vec::new()), None),
            next: 0,
            digest: Sha256::new(),
        };
        unpack(&target, &mut ordered, size)?;
        anyhow::Ok(hex::encode(ordered.digest.finalize()))
    });
    let fetched = stream::iter((0..size).step_by(cache.chunk as usize))
        .then(|offset| reserve(cache, size, offset))
        .map(|(offset, reserved)| {
            fetch(http, run_id, input, info, offset, reserved, sender.clone())
        })
        .buffer_unordered(1024)
        .try_collect::<()>()
        .await;
    drop(sender);
    let actual = unpacking.await?;
    fetched?;
    let actual = actual?;
    ensure!(
        info.sha256
            .as_ref()
            .is_none_or(|expected| *expected == actual),
        "downloaded asset hash does not match metadata"
    );
    tokio::fs::rename(&partial, entry).await?;
    Ok(())
}

async fn reserve(
    cache: &Cache,
    size: u64,
    offset: u64,
) -> (u64, anyhow::Result<OwnedSemaphorePermit>) {
    let length = cache.chunk.min(size - offset) as u32;
    let reserved = cache.memory.clone().acquire_many_owned(length).await;
    (offset, reserved.map_err(anyhow::Error::from))
}

async fn fetch(
    http: &HttpClient,
    run_id: &str,
    input: &str,
    info: &AssetDownload,
    offset: u64,
    reserved: anyhow::Result<OwnedSemaphorePermit>,
    chunks: mpsc::Sender<Chunk>,
) -> anyhow::Result<()> {
    let reserved = reserved?;
    let length = reserved.num_permits() as u64;
    let range = format!("bytes={offset}-{}", offset + length - 1);
    let (status, _, bytes) = http
        .request(
            Method::GET,
            &["runs", run_id, "inputs", input, "bytes"],
            None,
            &[("range", range), ("if-match", info.etag.clone())],
        )
        .await?;
    ensure!(
        status == StatusCode::PARTIAL_CONTENT && bytes.len() as u64 == length,
        "server returned an unexpected asset range"
    );
    chunks.send((offset, bytes, reserved))?;
    Ok(())
}

fn lock(path: &Path) -> io::Result<File> {
    loop {
        let file = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path)?;
        file.lock_shared()?;
        let (held, current) = (file.metadata()?, std::fs::metadata(path));
        if current.is_ok_and(|current| current.ino() == held.ino() && current.dev() == held.dev()) {
            return Ok(file);
        }
    }
}

fn evict(dir: &Path, limit: u64, incoming: u64) -> io::Result<()> {
    let mut entries = Vec::new();
    for item in std::fs::read_dir(dir)?.flatten() {
        let path = item.path();
        if let Ok(used) = std::fs::metadata(&path).and_then(|m| m.modified())
            && path.is_dir()
        {
            entries.push((used, disk_usage(&path), path));
        }
    }
    entries.sort();
    let mut total: u64 = entries.iter().map(|(_, size, _)| size).sum();
    let mut free = fs2::available_space(dir)?;
    let reserve = fs2::total_space(dir)? / 20;
    for (_, size, path) in entries {
        if total.saturating_add(incoming) <= limit && free >= incoming.saturating_add(reserve) {
            break;
        }
        let lock_path = path.with_extension("lock");
        let Ok(lock) = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)
        else {
            continue;
        };
        if lock.try_lock().is_err() || std::fs::remove_dir_all(&path).is_err() {
            continue;
        }
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
        .flatten()
        .map(|child| disk_usage(&child.path()))
        .sum()
}

struct Ordered {
    chunks: mpsc::Receiver<Chunk>,
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
    let checksum: u32 = block[..148]
        .iter()
        .chain(&block[156..])
        .map(|&b| u32::from(b))
        .sum();
    let header = tar::Header::from_byte_slice(&block);
    let is_tar = head.len() == 512
        && (header.cksum().ok() == Some(checksum + 8 * 32) || (block == [0; 512] && size >= 1024));
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
        let kind = entry.header().entry_type();
        ensure!(
            kind.is_file() || kind.is_dir(),
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
        ensure!(
            selected.starts_with(extracted.canonicalize()?),
            "asset entrypoint escapes extraction directory"
        );
        return Ok(selected);
    }
    let children = std::fs::read_dir(&extracted)?.collect::<Result<Vec<_>, _>>()?;
    match children.as_slice() {
        [child] => Ok(child.path()),
        _ => Ok(extracted),
    }
}
