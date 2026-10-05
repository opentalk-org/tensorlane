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
                    || name.ends_with(".plan")
                    || name
                        .split_once('-')
                        .is_some_and(|(a, b)| a.parse::<u64>().is_ok() && b.parse::<u64>().is_ok());
                if reusable {
                    let mut size = metadata.len();
                    if name.ends_with(".plan") {
                        for extension in ["index", "ready", "error"] {
                            match std::fs::metadata(entry.path().with_extension(extension)) {
                                Ok(metadata) => size += metadata.len(),
                                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                                Err(error) => return Err(error),
                            }
                        }
                    }
                    entries.push(Entry {
                        path: entry.path(),
                        size,
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
        let plan = entry.path.extension().is_some_and(|ext| ext == "plan");
        let lock_path = if plan || entry.path.extension().is_some_and(|ext| ext == "batch") {
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
        let paths = std::iter::once(entry.path.clone()).chain(
            ["index", "ready", "error"]
                .into_iter()
                .filter(|_| plan)
                .map(|extension| entry.path.with_extension(extension)),
        );
        for path in paths {
            match fs::remove_file(path).await {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        used = used.saturating_sub(entry.size);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn pruning_clears_query_plans_and_preserves_receipts_and_locked_entries() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let dir = temp.path().join("assets/a");
        fs::create_dir_all(&dir).await?;
        fs::write(dir.join("0-3"), b"abcd").await?;
        let run = temp.path().join("runs/r");
        let plans = run.join("plans");
        let data = run.join("data");
        fs::create_dir_all(&plans).await?;
        fs::create_dir_all(&data).await?;
        fs::write(data.join("0.batch"), b"data").await?;
        fs::write(plans.join("0.plan"), b"plan").await?;
        fs::write(plans.join("0.index"), b"index").await?;
        fs::write(plans.join("0.ready"), b"ready").await?;
        fs::write(plans.join("0.error"), b"error").await?;
        fs::write(run.join("receipt.json"), b"receipt").await?;
        let lock = Lock::acquire(&dir.join("asset.lock")).await?;
        let plan_lock = Lock::acquire(&plans.join("0.lock")).await?;
        prune(temp.path(), 0).await?;
        assert!(dir.join("0-3").exists());
        assert!(plans.join("0.plan").exists());
        assert!(plans.join("0.index").exists());
        assert!(plans.join("0.ready").exists());
        assert!(plans.join("0.error").exists());
        assert!(!data.join("0.batch").exists());
        drop(lock);
        drop(plan_lock);
        prune(temp.path(), 0).await?;
        assert!(!dir.join("0-3").exists());
        for extension in ["plan", "index", "ready", "error"] {
            assert!(!plans.join(format!("0.{extension}")).exists());
        }
        assert!(run.join("receipt.json").exists());
        Ok(())
    }
}
