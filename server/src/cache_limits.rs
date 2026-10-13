use crate::shared_cache::Lock;
use anyhow::{Result, ensure};
use std::{
    collections::HashMap,
    ffi::OsStr,
    io::ErrorKind,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::SystemTime,
};
use tokio::fs;

type Group = (SystemTime, u64, Vec<PathBuf>);

pub async fn space(path: &Path, bytes: u64) -> Result<()> {
    let path = path.to_owned();
    let available = tokio::task::spawn_blocking(move || fs2::available_space(path)).await??;
    ensure!(
        available >= bytes,
        "shared cache has insufficient free space"
    );
    Ok(())
}

pub async fn reserve(root: &Path, limit: u64, written: &AtomicU64, bytes: u64) -> Result<()> {
    if written.fetch_add(bytes, Ordering::Relaxed) + bytes > limit / 16 {
        prune(root, limit, written).await?;
    }
    Ok(())
}

pub async fn prune(root: &Path, limit: u64, written: &AtomicU64) -> Result<()> {
    let Some(_lock) = Lock::try_acquire(&root.join("prune.lock")).await? else {
        return Ok(());
    };
    written.store(0, Ordering::Relaxed);
    let root = root.to_owned();
    let mut groups = tokio::task::spawn_blocking(move || groups(&root)).await??;
    let mut used: u64 = groups.iter().map(|(_, size, _)| size).sum();
    groups.sort();
    for (_, size, mut members) in groups {
        if used <= limit {
            break;
        }
        let lock = members[0].with_extension("lock");
        let Some(_lock) = Lock::try_acquire(&lock).await? else {
            continue;
        };
        members.push(lock);
        members.sort_by_key(|path| {
            (
                path.extension() != Some(OsStr::new("ready")),
                path.extension() == Some(OsStr::new("lock")),
            )
        });
        for member in members {
            match fs::remove_file(member).await {
                Err(error) if error.kind() != ErrorKind::NotFound => return Err(error.into()),
                _ => {}
            }
        }
        used -= size;
    }
    Ok(())
}

fn groups(root: &Path) -> std::io::Result<Vec<Group>> {
    let mut pending = vec![root.join("assets"), root.join("runs")];
    let mut groups: HashMap<PathBuf, Group> = HashMap::new();
    while let Some(path) = pending.pop() {
        let dir = match std::fs::read_dir(&path) {
            Ok(dir) => dir,
            Err(error) if error.kind() == ErrorKind::NotFound => continue,
            Err(error) => return Err(error),
        };
        for entry in dir {
            let path = entry?.path();
            let Ok(metadata) = std::fs::symlink_metadata(&path) else {
                continue;
            };
            if metadata.is_dir() {
                pending.push(path);
            } else if path.extension() != Some(OsStr::new("part")) {
                let group = groups.entry(path.with_extension("")).or_insert((
                    SystemTime::UNIX_EPOCH,
                    0,
                    Vec::new(),
                ));
                group.0 = group.0.max(metadata.modified()?);
                group.1 += metadata.len();
                group.2.push(path);
            }
        }
    }
    Ok(groups.into_values().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn pruning_removes_whole_unlocked_groups_and_keeps_locked_ones() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let range = temp.path().join("assets/a");
        let plans = temp.path().join("runs/r/plans-v2");
        let data = temp.path().join("runs/r/data/0/g");
        let inputs = temp.path().join("runs/r/inputs");
        for dir in [&range, &plans, &data, &inputs] {
            fs::create_dir_all(dir).await?;
        }
        fs::write(range.join("0-3"), b"abcd").await?;
        for name in ["0.plan", "0.index", "0.ready", "0.error", "0.index.x.part"] {
            fs::write(plans.join(name), b"plan").await?;
        }
        for name in ["0.batch", "0.lock", "0.error"] {
            fs::write(data.join(name), b"data").await?;
        }
        fs::write(inputs.join("h.json"), b"{}").await?;
        let range_lock = Lock::acquire(&range.join("0-3.lock")).await?;
        let plan_lock = Lock::acquire(&plans.join("0.lock")).await?;
        let written = AtomicU64::new(0);
        prune(temp.path(), 0, &written).await?;
        assert!(range.join("0-3").exists());
        for name in ["0.plan", "0.index", "0.ready", "0.error"] {
            assert!(plans.join(name).exists());
        }
        assert_eq!(std::fs::read_dir(&data)?.count(), 0);
        assert_eq!(std::fs::read_dir(&inputs)?.count(), 0);
        drop(range_lock);
        drop(plan_lock);
        prune(temp.path(), 0, &written).await?;
        assert_eq!(std::fs::read_dir(&range)?.count(), 0);
        let left: Vec<_> = std::fs::read_dir(&plans)?
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(left, ["0.index.x.part"]);
        Ok(())
    }

    #[tokio::test]
    async fn pruning_evicts_least_recently_used_groups_first() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let plans = temp.path().join("runs/r/plans-v2");
        let data = temp.path().join("runs/r/data/0/g");
        fs::create_dir_all(&plans).await?;
        fs::create_dir_all(&data).await?;
        fs::write(plans.join("0.plan"), b"plan").await?;
        fs::write(plans.join("0.ready"), b"ready").await?;
        fs::write(data.join("0.batch"), b"data").await?;
        std::fs::File::open(data.join("0.batch"))?
            .set_modified(SystemTime::now() - Duration::from_secs(60))?;
        prune(temp.path(), 9, &AtomicU64::new(0)).await?;
        assert!(!data.join("0.batch").exists());
        assert!(plans.join("0.plan").exists());
        for name in ["0.plan", "0.ready"] {
            std::fs::File::open(plans.join(name))?
                .set_modified(SystemTime::now() - Duration::from_secs(60))?;
        }
        fs::write(data.join("1.batch"), b"data").await?;
        prune(temp.path(), 4, &AtomicU64::new(0)).await?;
        assert!(!plans.join("0.plan").exists());
        assert!(!plans.join("0.ready").exists());
        assert!(data.join("1.batch").exists());
        Ok(())
    }

    #[tokio::test]
    async fn writes_prune_once_they_exceed_the_slack() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let data = temp.path().join("runs/r/data");
        fs::create_dir_all(&data).await?;
        fs::write(data.join("0.batch"), vec![0; 100]).await?;
        let written = AtomicU64::new(0);
        reserve(temp.path(), 32, &written, 2).await?;
        assert!(data.join("0.batch").exists());
        reserve(temp.path(), 32, &written, 1).await?;
        assert!(!data.join("0.batch").exists());
        assert_eq!(written.load(Ordering::Relaxed), 0);
        Ok(())
    }
}
