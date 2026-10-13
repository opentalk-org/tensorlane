use crate::shared_cache::Lock;
use anyhow::{Result, ensure};
use std::{
    collections::HashMap,
    ffi::OsStr,
    io::ErrorKind,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, SystemTime},
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

pub struct Budget {
    root: PathBuf,
    limit: u64,
    usage: Mutex<(u64, u64)>,
}

pub struct Reservation {
    budget: Arc<Budget>,
    bytes: u64,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let mut usage = self.budget.usage.lock().unwrap();
        usage.0 += self.bytes;
        usage.1 -= self.bytes;
    }
}

impl Budget {
    pub fn new(root: PathBuf, limit: u64) -> Arc<Self> {
        Arc::new(Self {
            root,
            limit,
            usage: Mutex::new((0, 0)),
        })
    }

    pub async fn open(root: PathBuf, limit: u64) -> Result<Arc<Self>> {
        let scan = root.clone();
        let (_, parts) = tokio::task::spawn_blocking(move || groups(&scan)).await??;
        for part in parts {
            let _ = fs::remove_file(part).await;
        }
        let budget = Self::new(root, limit);
        budget.prune(0).await?;
        Ok(budget)
    }

    pub async fn reserve(self: &Arc<Self>, bytes: u64) -> Result<Reservation> {
        ensure!(
            bytes <= self.limit,
            "cache entry of {bytes} bytes exceeds CACHE_BYTES"
        );
        loop {
            {
                let mut usage = self.usage.lock().unwrap();
                if usage.0 + usage.1 + bytes <= self.limit {
                    usage.1 += bytes;
                    return Ok(Reservation {
                        budget: self.clone(),
                        bytes,
                    });
                }
            }
            if !self.prune(bytes).await? {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }

    pub async fn prune(&self, incoming: u64) -> Result<bool> {
        let _lock = Lock::acquire(&self.root.join("prune.lock")).await?;
        let root = self.root.clone();
        let (mut groups, _) = tokio::task::spawn_blocking(move || groups(&root)).await??;
        let mut used: u64 = groups.iter().map(|(_, size, _)| size).sum();
        let target = self
            .limit
            .saturating_sub(self.usage.lock().unwrap().1 + incoming);
        groups.sort();
        for (_, size, mut members) in groups {
            if used <= target {
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
        self.usage.lock().unwrap().0 = used;
        Ok(used <= target)
    }
}

fn groups(root: &Path) -> std::io::Result<(Vec<Group>, Vec<PathBuf>)> {
    let mut pending = vec![root.join("assets"), root.join("runs")];
    let mut groups: HashMap<PathBuf, Group> = HashMap::new();
    let mut parts = Vec::new();
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
            } else if path.extension() == Some(OsStr::new("part")) {
                parts.push(path);
            } else {
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
    Ok((groups.into_values().collect(), parts))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let budget = Budget::new(temp.path().to_owned(), 0);
        budget.prune(0).await?;
        assert!(range.join("0-3").exists());
        for name in ["0.plan", "0.index", "0.ready", "0.error"] {
            assert!(plans.join(name).exists());
        }
        assert_eq!(std::fs::read_dir(&data)?.count(), 0);
        assert_eq!(std::fs::read_dir(&inputs)?.count(), 0);
        drop(range_lock);
        drop(plan_lock);
        budget.prune(0).await?;
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
        Budget::new(temp.path().to_owned(), 9).prune(0).await?;
        assert!(!data.join("0.batch").exists());
        assert!(plans.join("0.plan").exists());
        for name in ["0.plan", "0.ready"] {
            std::fs::File::open(plans.join(name))?
                .set_modified(SystemTime::now() - Duration::from_secs(60))?;
        }
        fs::write(data.join("1.batch"), b"data").await?;
        Budget::new(temp.path().to_owned(), 4).prune(0).await?;
        assert!(!plans.join("0.plan").exists());
        assert!(!plans.join("0.ready").exists());
        assert!(data.join("1.batch").exists());
        Ok(())
    }

    #[tokio::test]
    async fn reservations_never_exceed_the_limit_and_evict_old_entries() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let data = temp.path().join("runs/r/data");
        fs::create_dir_all(&data).await?;
        fs::write(data.join("0.batch"), vec![0; 20]).await?;
        fs::write(data.join("1.batch.x.part"), vec![0; 20]).await?;
        let budget = Budget::open(temp.path().to_owned(), 32).await?;
        assert!(!data.join("1.batch.x.part").exists());
        let first = budget.reserve(20).await?;
        assert!(!data.join("0.batch").exists());
        let waiting = tokio::spawn({
            let budget = budget.clone();
            async move { budget.reserve(20).await.map(drop) }
        });
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(!waiting.is_finished());
        drop(first);
        tokio::time::timeout(Duration::from_secs(2), waiting).await???;
        assert!(budget.reserve(33).await.is_err());
        Ok(())
    }
}
