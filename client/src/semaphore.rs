use anyhow::{Context, Result, ensure};
use sem_safe::named::{OpenFlags, Semaphore};
use std::{
    ffi::CString,
    io,
    mem::ManuallyDrop,
    sync::atomic::{AtomicBool, Ordering},
};

pub struct PosixSemaphore {
    handle: ManuallyDrop<Semaphore>,
    name: CString,
    owner: bool,
}

impl PosixSemaphore {
    pub fn create(capacity: usize) -> Result<Self> {
        ensure!(capacity > 0, "semaphore capacity must be positive");
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
    cancelled: AtomicBool,
}

impl BatchBudget {
    pub fn new(capacity: usize) -> Result<Self> {
        Ok(Self {
            semaphore: PosixSemaphore::create(capacity)?,
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
        if !self.cancelled.swap(true, Ordering::AcqRel) {
            let _ = self.semaphore.post();
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
        let budget = Arc::new(BatchBudget::new(1)?);
        assert!(budget.acquire()?);
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
        let budget = Arc::new(BatchBudget::new(1)?);
        let name = budget.semaphore.name()?.to_owned();
        assert!(budget.acquire()?);
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
}
