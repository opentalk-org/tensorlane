use anyhow::{Context, Result};
use sem_safe::named::{OpenFlags, Semaphore};
use std::{
    ffi::CString,
    io,
    mem::ManuallyDrop,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

pub struct PosixSemaphore {
    handle: ManuallyDrop<Semaphore>,
    name: CString,
    owner: bool,
}

impl PosixSemaphore {
    pub fn create(capacity: usize) -> Result<Self> {
        let capacity = u32::try_from(capacity).context("semaphore capacity too large")?;
        let identifier = uuid::Uuid::new_v4().simple().to_string();
        let name = CString::new(format!("/tl-{}", &identifier[..20]))?;
        Self::open_named(
            name,
            OpenFlags::Create {
                exclusive: true,
                mode: 0o600,
                value: capacity,
            },
            true,
        )
    }

    pub fn open(name: &str) -> Result<Self> {
        let name = CString::new(name)?;
        Self::open_named(name, OpenFlags::AccessOnly, false)
    }

    fn open_named(name: CString, flags: OpenFlags, owner: bool) -> Result<Self> {
        let handle = Semaphore::open(&name, flags)
            .map_err(|()| io::Error::last_os_error())
            .context("opening POSIX semaphore")?;
        Ok(Self {
            handle: ManuallyDrop::new(handle),
            name,
            owner,
        })
    }

    pub fn name(&self) -> Result<&str> {
        Ok(self.name.to_str()?)
    }

    pub fn wait(&self) -> Result<()> {
        loop {
            if self.handle.sem_ref().wait().is_ok() {
                return Ok(());
            }
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(error).context("waiting for a batch slot");
            }
        }
    }

    pub fn post(&self) -> Result<()> {
        self.handle
            .sem_ref()
            .post()
            .map_err(|()| io::Error::last_os_error())
            .context("releasing a batch slot")
    }
}

impl Drop for PosixSemaphore {
    fn drop(&mut self) {
        unsafe {
            let handle = ManuallyDrop::take(&mut self.handle);
            let _ = handle.close();
        }
        if self.owner {
            let _ = Semaphore::unlink(&self.name);
        }
    }
}

pub struct BatchBudget {
    pub semaphore: PosixSemaphore,
    pub capacity: usize,
    cancelled: AtomicBool,
    pub memory: Arc<MemoryBudget>,
}

impl BatchBudget {
    pub fn new(capacity: usize, memory_bytes: usize) -> Result<Self> {
        Ok(Self {
            memory: Arc::new(MemoryBudget::new(memory_bytes)?),
            semaphore: PosixSemaphore::create(0)?,
            capacity,
            cancelled: AtomicBool::new(false),
        })
    }

    pub fn acquire(&self) -> Result<bool> {
        if self.cancelled.load(Ordering::Acquire) {
            return Ok(false);
        }
        self.semaphore.wait()?;
        Ok(!self.cancelled.load(Ordering::Acquire))
    }

    pub fn cancel(&self) {
        self.memory.cancel();
        if !self.cancelled.swap(true, Ordering::AcqRel) {
            let _ = self.semaphore.post();
        }
    }
}

pub const MEMORY_UNIT: usize = 1024 * 1024;

pub struct MemoryBudget {
    pub semaphore: PosixSemaphore,
    capacity: usize,
    next: tokio::sync::watch::Sender<u64>,
    cancelled: AtomicBool,
}

pub struct MemoryLease {
    memory: Arc<MemoryBudget>,
    pub units: usize,
}

impl MemoryBudget {
    fn new(bytes: usize) -> Result<Self> {
        let capacity = bytes.div_ceil(MEMORY_UNIT).max(1);
        Ok(Self {
            semaphore: PosixSemaphore::create(capacity)?,
            capacity,
            next: tokio::sync::watch::channel(0).0,
            cancelled: AtomicBool::new(false),
        })
    }

    pub async fn acquire(self: &Arc<Self>, sequence: u64, bytes: usize) -> Result<MemoryLease> {
        anyhow::ensure!(
            !self.cancelled.load(Ordering::Acquire),
            "memory budget cancelled"
        );
        let mut next = self.next.subscribe();
        while *next.borrow_and_update() != sequence {
            anyhow::ensure!(
                !self.cancelled.load(Ordering::Acquire),
                "memory budget cancelled"
            );
            next.changed().await?;
        }
        let units = bytes.div_ceil(MEMORY_UNIT).min(self.capacity);
        let memory = self.clone();
        tokio::task::spawn_blocking(move || {
            for _ in 0..units {
                memory.semaphore.wait()?;
                anyhow::ensure!(
                    !memory.cancelled.load(Ordering::Acquire),
                    "memory budget cancelled"
                );
            }
            anyhow::Ok(())
        })
        .await??;
        anyhow::ensure!(
            !self.cancelled.load(Ordering::Acquire),
            "memory budget cancelled"
        );
        self.next.send_replace(sequence + 1);
        Ok(MemoryLease {
            memory: self.clone(),
            units,
        })
    }

    fn cancel(&self) {
        if !self.cancelled.swap(true, Ordering::AcqRel) {
            self.next.send_replace(u64::MAX);
            for _ in 0..self.capacity {
                let _ = self.semaphore.post();
            }
        }
    }
}

impl MemoryLease {
    pub fn transfer(mut self) {
        self.units = 0;
    }
}

impl Drop for MemoryLease {
    fn drop(&mut self) {
        for _ in 0..self.units {
            let _ = self.memory.semaphore.post();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::{Arc, mpsc},
        thread,
        time::Duration,
    };

    #[test]
    fn independently_opened_handle_releases_waiter() -> Result<()> {
        let budget = Arc::new(BatchBudget::new(1, 1024 * 1024)?);
        let rank = PosixSemaphore::open(budget.semaphore.name()?)?;
        let (done, received) = mpsc::channel();
        let waiting = budget.clone();
        let thread = thread::spawn(move || done.send(waiting.acquire()).unwrap());
        assert!(received.recv_timeout(Duration::from_millis(50)).is_err());
        rank.post()?;
        assert!(received.recv_timeout(Duration::from_secs(2))??);
        thread.join().unwrap();
        Ok(())
    }

    #[test]
    fn cancellation_wakes_waiter_and_unlinks_on_drop() -> Result<()> {
        let budget = Arc::new(BatchBudget::new(1, 1024 * 1024)?);
        let name = budget.semaphore.name()?.to_owned();
        let (done, received) = mpsc::channel();
        let waiting = budget.clone();
        let thread = thread::spawn(move || done.send(waiting.acquire()).unwrap());
        assert!(received.recv_timeout(Duration::from_millis(50)).is_err());
        budget.cancel();
        assert!(!received.recv_timeout(Duration::from_secs(2))??);
        thread.join().unwrap();
        drop(budget);
        assert!(PosixSemaphore::open(&name).is_err());
        Ok(())
    }

    #[test]
    fn closing_one_handle_preserves_other_handles() -> Result<()> {
        let owner = PosixSemaphore::create(1)?;
        let first = PosixSemaphore::open(owner.name()?)?;
        let second = PosixSemaphore::open(owner.name()?)?;
        drop(first);
        second.wait()?;
        owner.post()?;
        second.wait()?;
        Ok(())
    }

    #[test]
    fn unlink_preserves_already_open_handles() -> Result<()> {
        let owner = PosixSemaphore::create(1)?;
        let name = owner.name()?.to_owned();
        let rank = PosixSemaphore::open(&name)?;
        drop(owner);
        assert!(PosixSemaphore::open(&name).is_err());
        rank.wait()?;
        rank.post()?;
        Ok(())
    }

    #[tokio::test]
    async fn memory_reservations_preserve_sequence_and_allow_oversized_batches() -> Result<()> {
        let memory = Arc::new(MemoryBudget::new(2 * MEMORY_UNIT)?);
        let later = {
            let memory = memory.clone();
            tokio::spawn(async move { memory.acquire(1, 4 * MEMORY_UNIT).await })
        };
        let first = memory.acquire(0, MEMORY_UNIT).await?;
        assert_eq!(first.units, 1);
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!later.is_finished());
        drop(first);
        let oversized = tokio::time::timeout(Duration::from_secs(2), later).await???;
        assert_eq!(oversized.units, 2);
        let following = {
            let memory = memory.clone();
            tokio::spawn(async move { memory.acquire(2, MEMORY_UNIT).await })
        };
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!following.is_finished());
        drop(oversized);
        let _last = tokio::time::timeout(Duration::from_secs(2), following).await???;
        Ok(())
    }

    #[tokio::test]
    async fn cancellation_wakes_memory_and_sequence_waiters() -> Result<()> {
        let memory = Arc::new(MemoryBudget::new(MEMORY_UNIT)?);
        let _first = memory.acquire(0, MEMORY_UNIT).await?;
        let mut waiters = Vec::new();
        for sequence in [1, 2] {
            let memory = memory.clone();
            waiters.push(tokio::spawn(async move {
                memory.acquire(sequence, MEMORY_UNIT).await
            }));
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(waiters.iter().all(|waiter| !waiter.is_finished()));
        memory.cancel();
        for waiter in waiters {
            assert!(
                tokio::time::timeout(Duration::from_secs(2), waiter)
                    .await??
                    .is_err()
            );
        }
        Ok(())
    }
}
