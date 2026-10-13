use anyhow::{Context, Result};
use fs2::FileExt;
use std::{
    fs::File,
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::{fs, io::AsyncWriteExt};
use uuid::Uuid;

pub struct Lock {
    _file: File,
}
impl Lock {
    pub async fn try_acquire(path: &Path) -> Result<Option<Self>> {
        let path = path.to_owned();
        Ok(tokio::task::spawn_blocking(move || try_lock(&path)).await??)
    }
    pub async fn acquire(path: &Path) -> Result<Self> {
        loop {
            if let Some(lock) = Self::try_acquire(path).await? {
                return Ok(lock);
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}
fn try_lock(path: &Path) -> std::io::Result<Option<Lock>> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    match file.try_lock_exclusive() {
        Ok(()) => Ok(Some(Lock { _file: file })),
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Ok(None),
        Err(error) => Err(error),
    }
}
pub struct TemporaryFile(pub PathBuf);
impl Drop for TemporaryFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

pub async fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .context("cache file has no parent")?
        .to_owned();
    crate::cache_limits::space(&parent, bytes.len() as u64).await?;
    let part = TemporaryFile(path.with_extension(format!("{}.part", Uuid::new_v4())));
    let mut file = fs::File::create(&part.0).await?;
    file.write_all(bytes).await?;
    file.sync_all().await?;
    drop(file);
    fs::rename(&part.0, path)
        .await
        .context("publishing shared cache file")?;
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
