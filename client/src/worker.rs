mod execution;
use anyhow::{Context, anyhow, bail, ensure};
use fs2::FileExt;
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs::{File, OpenOptions, Permissions},
    num::NonZeroUsize,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    sync::{
        Arc, OnceLock,
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
#[derive(Deserialize)]
struct Config {
    #[serde(default)]
    tensorlane: Tunables,
}
#[derive(Deserialize, Default)]
#[serde(default)]
struct Tunables {
    ranks: Option<NonZeroUsize>,
    prefetch_factor: Option<NonZeroUsize>,
    num_workers: Option<NonZeroUsize>,
    max_prefetch_memory_bytes: Option<NonZeroUsize>,
}
impl Settings {
    fn resolve(options: &Options, config: &str) -> anyhow::Result<Self> {
        let config: Config = serde_json::from_str(config)
            .context("config.tensorlane settings must be positive integers")?;
        let tunables = config.tensorlane;
        let settings = Self {
            ranks: options
                .ranks
                .or(tunables.ranks.map(NonZeroUsize::get))
                .unwrap_or(1),
            factor: options
                .factor
                .or(tunables.prefetch_factor.map(NonZeroUsize::get))
                .unwrap_or(32),
            num_workers: options
                .num_workers
                .or(tunables.num_workers.map(NonZeroUsize::get))
                .unwrap_or(5),
            memory_bytes: tunables
                .max_prefetch_memory_bytes
                .map_or(2 * 1024 * 1024 * 1024, NonZeroUsize::get),
        };
        ensure!(options.rank < settings.ranks, "invalid rank");
        Ok(settings)
    }
}
pub struct Worker {
    stop: Option<oneshot::Sender<anyhow::Result<()>>>,
    thread: Option<JoinHandle<()>>,
    connected: Arc<AtomicBool>,
    outcome: Arc<OnceLock<Result<(), String>>>,
    root: PathBuf,
    _lock: File,
}
impl Worker {
    pub fn start(options: Options) -> anyhow::Result<(Self, Initialized)> {
        let root = options.root.clone();
        std::fs::create_dir_all(&root)?;
        std::fs::set_permissions(&root, Permissions::from_mode(0o700))?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root.join("lock"))?;
        lock.try_lock_exclusive()
            .context("a daemon is already active for this run")?;
        clear(&root)?;
        let auth = [
            uuid::Uuid::new_v4().into_bytes(),
            uuid::Uuid::new_v4().into_bytes(),
        ];
        std::fs::write(root.join("auth"), auth.concat())?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()?;
        let (stop, stopped) = oneshot::channel();
        let (ready, initialized) = std::sync::mpsc::sync_channel(1);
        let connected = Arc::new(AtomicBool::new(false));
        let outcome = Arc::new(OnceLock::new());
        let (thread_connected, thread_outcome) = (connected.clone(), outcome.clone());
        let thread = thread::Builder::new()
            .name("tensorlane-daemon".into())
            .spawn(move || {
                let supervise = execution::supervise(options, stopped, ready, thread_connected);
                let result = runtime
                    .block_on(supervise)
                    .map_err(|error| format!("{error:#}"));
                if let Err(error) = &result {
                    eprintln!("TensorLane daemon failed: {error}");
                }
                let _ = thread_outcome.set(result);
            })?;
        let mut worker = Self {
            stop: Some(stop),
            thread: Some(thread),
            connected,
            outcome,
            root,
            _lock: lock,
        };
        let Ok(initialized) = initialized.recv() else {
            worker.shutdown()?;
            return Err(worker.check().expect_err("daemon thread has exited"));
        };
        Ok((worker, initialized))
    }
    pub fn stop(&mut self, error: Option<String>) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(error.map_or(Ok(()), |message| Err(anyhow!(message))));
        }
    }
    pub fn check(&self) -> anyhow::Result<()> {
        match self.outcome.get() {
            Some(Err(error)) => bail!("{error}"),
            Some(Ok(())) => bail!("TensorLane daemon is closed"),
            None => ensure!(
                self.thread
                    .as_ref()
                    .is_some_and(|thread| !thread.is_finished()),
                "TensorLane daemon stopped unexpectedly"
            ),
        }
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
        let _ = clear(&self.root);
    }
}
fn clear(root: &Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_name() == "lock" {
            continue;
        }
        if entry.file_type()?.is_dir() {
            std::fs::remove_dir_all(entry.path())?;
        } else {
            std::fs::remove_file(entry.path())?;
        }
    }
    Ok(())
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
            (1, 5, 32)
        );
    }
}
