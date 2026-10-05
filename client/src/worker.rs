use crate::semaphore::BatchBudget;
mod execution;
use anyhow::{Context, anyhow, ensure};
use fs2::FileExt;
use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::Write,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
};
use tensorlane_protocol::InitResponse;
use tokio::sync::oneshot;

pub struct Options {
    pub run_id: String,
    pub addr: String,
    pub api_key: Option<String>,
    pub root: PathBuf,
    pub ranks: Option<usize>,
    pub factor: Option<usize>,
    pub num_workers: Option<usize>,
    pub rank: usize,
    pub startup_timeout: Option<std::time::Duration>,
}
pub struct Initialized {
    pub response: InitResponse,
    pub assets: HashMap<String, PathBuf>,
    pub asset_metadata: HashMap<String, String>,
    pub settings: Settings,
}
#[derive(Clone, Copy)]
pub struct Settings {
    pub ranks: usize,
    pub factor: usize,
    pub num_workers: usize,
    pub memory_bytes: usize,
}
impl Settings {
    fn resolve(options: &Options, config: &str) -> anyhow::Result<Self> {
        let config: serde_json::Map<String, serde_json::Value> = serde_json::from_str(config)?;
        let settings = match config.get("tensorlane") {
            Some(value) => Some(
                value
                    .as_object()
                    .context("config.tensorlane must be an object")?,
            ),
            None => None,
        };
        let count =
            |name: &str, explicit: Option<usize>, default: usize| -> anyhow::Result<usize> {
                if let Some(value) = explicit {
                    return Ok(value);
                }
                match settings.and_then(|settings| settings.get(name)) {
                    Some(value) if !value.is_null() => {
                        let value =
                            value.as_u64().filter(|value| *value > 0).with_context(|| {
                                format!("config.tensorlane.{name} must be a positive integer")
                            })?;
                        Ok(usize::try_from(value)?)
                    }
                    _ => Ok(default),
                }
            };
        let settings = Self {
            ranks: count("ranks", options.ranks, 1)?,
            factor: count("prefetch_factor", options.factor, 2)?,
            num_workers: count("num_workers", options.num_workers, 5)?,
            memory_bytes: count("max_prefetch_memory_bytes", None, 128 * 1024 * 1024)?,
        };
        ensure!(options.rank < settings.ranks, "invalid rank");
        Ok(settings)
    }
}
pub struct Worker {
    stop: Option<oneshot::Sender<anyhow::Result<()>>>,
    thread: Option<JoinHandle<()>>,
    connected: Arc<AtomicBool>,
    outcome: Arc<Mutex<Option<Result<(), String>>>>,
    _resources: Resources,
}

struct Resources {
    root: PathBuf,
    budgets: Arc<Mutex<HashMap<String, Arc<BatchBudget>>>>,
    _lock: File,
}
impl Resources {
    fn new(root: &Path) -> anyhow::Result<Self> {
        std::fs::create_dir_all(root)?;
        std::fs::set_permissions(root, std::fs::Permissions::from_mode(0o700))?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root.join("lock"))?;
        lock.try_lock_exclusive()
            .context("a daemon is already active for this run")?;
        Self::clear(root)?;
        let mut auth = File::create(root.join("auth"))?;
        auth.write_all(uuid::Uuid::new_v4().as_bytes())?;
        auth.write_all(uuid::Uuid::new_v4().as_bytes())?;
        let budgets = Arc::new(Mutex::new(HashMap::new()));
        Ok(Self {
            root: root.to_owned(),
            budgets,
            _lock: lock,
        })
    }

    fn clear(root: &Path) -> std::io::Result<()> {
        for entry in std::fs::read_dir(root)? {
            let entry = entry?;
            if entry.file_name() != "lock" {
                if entry.file_type()?.is_dir() {
                    std::fs::remove_dir_all(entry.path())?;
                } else {
                    std::fs::remove_file(entry.path())?;
                }
            }
        }
        Ok(())
    }
}
impl Drop for Resources {
    fn drop(&mut self) {
        let _ = Self::clear(&self.root);
    }
}
impl Worker {
    pub fn start(options: Options) -> anyhow::Result<(Self, Initialized)> {
        for (name, value) in [
            ("ranks", options.ranks),
            ("prefetch_factor", options.factor),
            ("num_workers", options.num_workers),
        ] {
            ensure!(
                value.is_none_or(|value| value > 0),
                "{name} must be positive"
            );
        }
        let resources = Resources::new(&options.root)?;
        let session_path = options.root.join("session");
        let session = uuid::Uuid::new_v4().to_string();
        std::fs::write(session_path, &session)?;
        let budgets = resources.budgets.clone();
        let root = options.root.clone();
        let (stop, stopped) = oneshot::channel();
        let (ready, initialized) = std::sync::mpsc::sync_channel(1);
        let connected = Arc::new(AtomicBool::new(false));
        let outcome = Arc::new(Mutex::new(None));
        let thread_connected = connected.clone();
        let thread_outcome = outcome.clone();
        let thread = thread::Builder::new()
            .name("tensorlane-daemon".into())
            .spawn(move || {
                let result = (|| -> anyhow::Result<()> {
                    let runtime = tokio::runtime::Builder::new_multi_thread()
                        .worker_threads(2)
                        .enable_all()
                        .build()?;
                    runtime.block_on(execution::supervise(
                        options,
                        session,
                        budgets,
                        stopped,
                        &ready,
                        &thread_connected,
                    ))
                })();
                let _ = std::fs::remove_file(root.join("init.json"));
                if let Ok(mut outcome) = thread_outcome.lock() {
                    *outcome = Some(
                        result
                            .as_ref()
                            .map(|_| ())
                            .map_err(|error| format!("{error:#}")),
                    );
                }
                if let Err(error) = result {
                    let message = format!("{error:#}");
                    eprintln!("TensorLane daemon failed: {message}");
                    let _ = ready.send(Err(anyhow!(message)));
                }
            })?;
        let mut worker = Self {
            stop: Some(stop),
            thread: Some(thread),
            connected,
            outcome,
            _resources: resources,
        };
        match initialized
            .recv()
            .context("daemon stopped during startup")?
        {
            Ok(response) => Ok((worker, response)),
            Err(error) => {
                worker.shutdown()?;
                Err(error)
            }
        }
    }
    pub fn stop(&mut self, error: Option<String>) {
        let _ = std::fs::remove_file(self._resources.root.join("init.json"));
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(error.map_or(Ok(()), |message| Err(anyhow!(message))));
        }
        if let Ok(budgets) = self._resources.budgets.lock() {
            for budget in budgets.values() {
                budget.cancel();
            }
        }
    }
    pub fn check(&self) -> anyhow::Result<()> {
        let outcome = self
            .outcome
            .lock()
            .map_err(|_| anyhow!("daemon outcome lock poisoned"))?;
        match outcome.as_ref() {
            Some(Err(error)) => return Err(anyhow!(error.clone())),
            Some(Ok(())) => return Err(anyhow!("TensorLane daemon is closed")),
            None => {}
        }
        ensure!(
            self.thread
                .as_ref()
                .is_some_and(|thread| !thread.is_finished()),
            "TensorLane daemon stopped unexpectedly"
        );
        Ok(())
    }
    pub fn ready(&self) -> anyhow::Result<bool> {
        self.check()?;
        Ok(self.connected.load(Ordering::Acquire))
    }
    pub fn shutdown(&mut self) -> anyhow::Result<()> {
        self.stop(None);
        if let Some(thread) = self.thread.take() {
            thread
                .join()
                .map_err(|_| anyhow!("daemon thread panicked"))?;
        }
        Ok(())
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}
#[cfg(test)]
mod settings_tests {
    use super::{Options, Settings};

    fn options() -> Options {
        Options {
            run_id: "settings-test".into(),
            addr: String::new(),
            api_key: None,
            root: Default::default(),
            ranks: None,
            factor: None,
            num_workers: None,
            rank: 0,
            startup_timeout: None,
        }
    }

    #[test]
    fn nested_runtime_config_uses_eight_workers_and_prefetch_eight() {
        let config = r#"{
            "tensorlane": {"num_workers": 8, "prefetch_factor": 8},
            "app": {"num_workers": 99, "prefetch_factor": 99}
        }"#;
        let settings = Settings::resolve(&options(), config).unwrap();
        assert_eq!(
            (settings.ranks, settings.num_workers, settings.factor),
            (1, 8, 8)
        );
    }

    #[test]
    fn explicit_arguments_override_nested_runtime_config() {
        let mut options = options();
        options.ranks = Some(2);
        options.num_workers = Some(3);
        options.factor = Some(4);
        options.rank = 1;
        let config = r#"{"tensorlane": {"ranks": 1, "num_workers": 8, "prefetch_factor": 8}}"#;
        let settings = Settings::resolve(&options, config).unwrap();
        assert_eq!(
            (settings.ranks, settings.num_workers, settings.factor),
            (2, 3, 4)
        );
    }

    #[test]
    fn missing_runtime_config_uses_defaults_and_ignores_app_fields() {
        let config = r#"{"app": {"ranks": 2, "num_workers": 8, "prefetch_factor": 8}}"#;
        let settings = Settings::resolve(&options(), config).unwrap();
        assert_eq!(
            (settings.ranks, settings.num_workers, settings.factor),
            (1, 5, 2)
        );
    }
}
