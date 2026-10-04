use crate::shared_cache::Lock;
use anyhow::{Context, Result, ensure};
use std::{
    path::{Path, PathBuf},
    time::SystemTime,
};
use tokio::fs;

pub async fn space(path: &Path, bytes: u64) -> Result<()> {
    let path = path.to_owned();
    let available = tokio::task::spawn_blocking(move || fs2::available_space(path)).await??;
    ensure!(
        available >= bytes.saturating_add(512 * 1024 * 1024),
        "shared cache has insufficient free space"
    );
    Ok(())
}

struct Entry {
    path: PathBuf,
    size: u64,
    modified: SystemTime,
}

pub async fn prune(root: &Path, limit: u64) -> Result<()> {
    let Some(_lock) = Lock::try_acquire(&root.join("prune.lock")).await? else {
        return Ok(());
    };
    let root = root.to_owned();
    let mut entries = tokio::task::spawn_blocking(move || {
        let mut pending = vec![root.join("assets"), root.join("runs")];
        let mut entries = Vec::new();
        while let Some(path) = pending.pop() {
            let dir = match std::fs::read_dir(&path) {
                Ok(dir) => dir,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error),
            };
            for entry in dir {
                let entry = entry?;
                let metadata = entry.metadata()?;
                if metadata.is_dir() {
                    pending.push(entry.path());
                    continue;
                }
                let name = entry.file_name();
                let name = name.to_string_lossy();
                let reusable = name.ends_with(".batch")
                    || name
                        .split_once('-')
                        .is_some_and(|(a, b)| a.parse::<u64>().is_ok() && b.parse::<u64>().is_ok());
                if reusable {
                    entries.push(Entry {
                        path: entry.path(),
                        size: metadata.len(),
                        modified: metadata.modified()?,
                    });
                }
            }
        }
        Ok::<_, std::io::Error>(entries)
    })
    .await??;
    let mut used: u64 = entries.iter().map(|entry| entry.size).sum();
    entries.sort_by_key(|entry| entry.modified);
    for entry in entries {
        if used <= limit {
            break;
        }
        let lock_path = if entry.path.extension().is_some_and(|ext| ext == "batch") {
            entry.path.with_extension("lock")
        } else {
            entry
                .path
                .parent()
                .context("asset range has no directory")?
                .join("asset.lock")
        };
        let Some(_lock) = Lock::try_acquire(&lock_path).await? else {
            continue;
        };
        match fs::remove_file(entry.path).await {
            Ok(()) => used = used.saturating_sub(entry.size),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn pruning_preserves_plans_receipts_and_locked_ranges() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let dir = temp.path().join("assets/a");
        fs::create_dir_all(&dir).await?;
        fs::write(dir.join("0-3"), b"abcd").await?;
        let run = temp.path().join("runs/r");
        fs::create_dir_all(&run).await?;
        fs::write(run.join("0.batch"), b"data").await?;
        fs::write(run.join("0.plan"), b"plan").await?;
        let lock = Lock::acquire(&dir.join("asset.lock")).await?;
        prune(temp.path(), 0).await?;
        assert!(dir.join("0-3").exists());
        assert!(run.join("0.plan").exists());
        assert!(!run.join("0.batch").exists());
        drop(lock);
        prune(temp.path(), 0).await?;
        assert!(!dir.join("0-3").exists());
        Ok(())
    }
}
