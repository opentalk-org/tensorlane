use anyhow::{Context, Result};
use fs2::FileExt;
use std::{
    fs::File,
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::{fs, io::AsyncWriteExt};
use uuid::Uuid;

pub struct Lock(File);
impl Lock {
    pub async fn try_acquire(path: &Path) -> Result<Option<Self>> {
        let path = path.to_owned();
        tokio::task::spawn_blocking(move || {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(path)?;
            match file.try_lock_exclusive() {
                Ok(()) => Ok(Some(Self(file))),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(None),
                Err(e) => Err(e.into()),
            }
        })
        .await?
    }
    pub async fn acquire(path: &Path) -> Result<Self> {
        let path = path.to_owned();
        let file = tokio::task::spawn_blocking(move || {
            std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(path)
        })
        .await??;
        loop {
            match file.try_lock_exclusive() {
                Ok(()) => return Ok(Self(file)),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                Err(e) => return Err(e.into()),
            }
        }
    }
}
impl Drop for Lock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.0);
    }
}

pub struct TemporaryFile(pub PathBuf);
impl Drop for TemporaryFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

pub async fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    crate::cache_limits::space(
        path.parent().context("cache file has no parent")?,
        bytes.len() as u64,
    )
    .await?;
    let part = TemporaryFile(path.with_extension(format!("{}.part", Uuid::new_v4())));
    let mut file = fs::File::create(&part.0).await?;
    file.write_all(bytes).await?;
    file.sync_all().await?;
    drop(file);
    fs::rename(&part.0, path)
        .await
        .context("publishing shared cache file")?;
    let parent = path
        .parent()
        .context("cache file has no parent")?
        .to_owned();
    tokio::task::spawn_blocking(move || File::open(parent)?.sync_all()).await??;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn lock_is_shared_between_independent_callers_and_released_on_drop() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("lock");
        let first = Lock::acquire(&path).await?;
        assert!(
            tokio::time::timeout(Duration::from_millis(100), Lock::acquire(&path))
                .await
                .is_err()
        );
        drop(first);
        tokio::time::timeout(Duration::from_secs(1), Lock::acquire(&path)).await??;
        Ok(())
    }
}
