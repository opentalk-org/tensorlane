use crate::{
    data::{Work, prefetch},
    ipc::Sender,
    proto::{EndRequest, InitRequest, InitResponse, tensor_lane_client::TensorLaneClient},
    semaphore::BatchBudget,
};
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
    time::Duration,
};
use tokio::{
    net::UnixListener,
    sync::{mpsc, oneshot},
};
use tokio_util::sync::CancellationToken;
use tonic::transport::{Channel, Endpoint};

pub struct Options {
    pub run_id: String,
    pub addr: String,
    pub root: PathBuf,
    pub ranks: Option<usize>,
    pub factor: Option<usize>,
    pub num_workers: Option<usize>,
    pub rank: usize,
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
}
impl Settings {
    fn resolve(options: &Options, config: &str) -> anyhow::Result<Self> {
        let config: serde_json::Map<String, serde_json::Value> = serde_json::from_str(config)?;
        let count =
            |name: &str, explicit: Option<usize>, default: usize| -> anyhow::Result<usize> {
                let value = match explicit {
                    Some(value) => value,
                    None => match config.get(name) {
                        Some(value) => usize::try_from(value.as_u64().with_context(|| {
                            format!("config.{name} must be a positive integer")
                        })?)?,
                        None => default,
                    },
                };
                ensure!(value > 0, "{name} must be positive");
                Ok(value)
            };
        let settings = Self {
            ranks: count("ranks", options.ranks, 1)?,
            factor: count("prefetch_factor", options.factor, 2)?,
            num_workers: count("num_workers", options.num_workers, 5)?,
        };
        ensure!(options.rank < settings.ranks, "invalid rank");
        u32::try_from(
            settings
                .ranks
                .checked_mul(settings.factor)
                .context("prefetch capacity overflow")?,
        )
        .context("prefetch capacity too large")?;
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
}
impl Drop for Resources {
    fn drop(&mut self) {
        if let Ok(entries) = std::fs::read_dir(&self.root) {
            for entry in entries.flatten() {
                if entry.file_name() != "lock" {
                    if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
                        let _ = std::fs::remove_dir_all(entry.path());
                    } else {
                        let _ = std::fs::remove_file(entry.path());
                    }
                }
            }
        }
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
                    runtime.block_on(supervise(
                        options,
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
        if let Ok(budgets) = self._resources.budgets.lock() {
            for budget in budgets.values() {
                budget.cancel();
            }
        }
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(error.map_or(Ok(()), |message| Err(anyhow!(message))));
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
async fn supervise(
    options: Options,
    budgets: Arc<Mutex<HashMap<String, Arc<BatchBudget>>>>,
    mut stop: oneshot::Receiver<anyhow::Result<()>>,
    ready: &std::sync::mpsc::SyncSender<anyhow::Result<Initialized>>,
    connected: &AtomicBool,
) -> anyhow::Result<()> {
    let work_listener = UnixListener::bind(options.root.join("work.sock"))?;
    let upload_listener = UnixListener::bind(options.root.join("uploads.sock"))?;
    let url = if options.addr.contains("://") {
        options.addr.clone()
    } else {
        format!("http://{}", options.addr)
    };
    let mut remote = None;
    let startup = async {
        let channel = Endpoint::from_shared(url)?
            .connect_timeout(Duration::from_secs(10))
            .connect()
            .await?;
        let mut grpc = TensorLaneClient::new(channel)
            .max_decoding_message_size(crate::MAX_BATCH_BYTES)
            .max_encoding_message_size(crate::MAX_BATCH_BYTES);
        let initialized = grpc
            .init(InitRequest {
                run_id: options.run_id.clone(),
            })
            .await?
            .into_inner();
        remote = Some((grpc.clone(), initialized.run_id.clone()));
        let settings = Settings::resolve(&options, &initialized.config)?;
        std::fs::write(options.root.join("ranks"), settings.ranks.to_string())?;
        ensure!(
            !initialized.streams.is_empty(),
            "server returned no streams"
        );
        let (assets, asset_metadata) =
            crate::assets::prefetch(&grpc, &initialized, &options.root).await?;
        {
            let mut budgets = budgets
                .lock()
                .map_err(|_| anyhow!("budget lock poisoned"))?;
            for (index, name) in initialized.streams.iter().enumerate() {
                ensure!(!budgets.contains_key(name), "duplicate stream name");
                let budget = Arc::new(BatchBudget::new(
                    settings
                        .ranks
                        .checked_mul(settings.factor)
                        .context("prefetch capacity overflow")?,
                )?);
                let directory = options.root.join("streams").join(index.to_string());
                std::fs::create_dir_all(&directory)?;
                std::fs::write(directory.join("semaphore"), budget.semaphore.name()?)?;
                budgets.insert(name.clone(), budget);
            }
        }
        ready
            .send(Ok(Initialized {
                response: initialized.clone(),
                assets,
                asset_metadata,
                settings,
            }))
            .map_err(|_| anyhow!("initializer disconnected"))?;
        let mut sockets = Vec::new();
        for _ in 0..settings.num_workers {
            sockets.push(work_listener.accept().await?.0);
        }
        let mut work = Vec::new();
        for socket in sockets {
            work.push(Sender::<Work, _>::new(socket));
        }
        anyhow::Ok((grpc, initialized, work))
    };
    let started = tokio::select! {
        result = tokio::time::timeout(Duration::from_mins(10), startup) => result.context("daemon startup timed out").and_then(|result| result).map(Some),
        result = &mut stop => result.unwrap_or(Ok(())).map(|()| None),
    };
    let (grpc, initialized, work) = match started {
        Ok(Some(started)) => started,
        result => {
            let ended = match remote {
                Some((grpc, run_id)) => end_run(grpc, run_id).await,
                None => Ok(()),
            };
            return result.map(|_| ()).and(ended);
        }
    };
    let uploads_stopping = CancellationToken::new();
    let mut uploads = tokio::spawn(crate::uploads::serve(
        upload_listener,
        grpc.clone(),
        initialized.run_id.clone(),
        uploads_stopping.clone(),
    ));
    let mut uploads_complete = false;
    let (send_work, mut receive_work) = mpsc::unbounded_channel::<Work>();
    let mut sender = tokio::spawn(async move {
        let mut senders = work;
        let mut next_worker = 0;
        while let Some(message) = receive_work.recv().await {
            match &message {
                Work::End { .. } => {
                    for sender in &mut senders {
                        sender
                            .send(&message)
                            .await
                            .context("transform worker disconnected")?;
                    }
                }
                Work::Sample { .. } => {
                    senders[next_worker]
                        .send(&message)
                        .await
                        .context("transform worker disconnected")?;
                    next_worker = (next_worker + 1) % senders.len();
                }
            }
        }
        anyhow::Ok(senders)
    });
    let mut pumps = tokio::task::JoinSet::new();
    let stream_budgets = budgets
        .lock()
        .map_err(|_| anyhow!("budget lock poisoned"))?
        .clone();
    for (name, budget) in stream_budgets {
        pumps.spawn(prefetch(
            grpc.clone(),
            initialized.run_id.clone(),
            name,
            budget,
            send_work.clone(),
        ));
    }
    drop(send_work);
    let mut idle_senders = None;
    let mut sender_complete = false;
    let result = async {
        connected.store(true, Ordering::Release);
        loop {
            tokio::select! {
                result = &mut stop => return result.unwrap_or(Ok(())),
                result = &mut uploads => {
                    uploads_complete = true;
                    result.context("upload task panicked")??;
                    return Err(anyhow!("upload listener stopped unexpectedly"));
                },
                Some(result) = pumps.join_next(), if !pumps.is_empty() => {
                    result.context("prefetch task panicked")??;
                },
                result = &mut sender, if !sender_complete => {
                    sender_complete = true;
                    idle_senders = Some(result.context("socket sender panicked")??);
                },
            }
        }
    }
    .await;
    let _ = std::fs::remove_file(options.root.join("init.json"));
    if let Ok(budgets) = budgets.lock() {
        for budget in budgets.values() {
            budget.cancel();
        }
    }
    pumps.abort_all();
    while pumps.join_next().await.is_some() {}
    drop(idle_senders);
    if !sender_complete {
        sender.abort();
        let _ = sender.await;
    }
    uploads_stopping.cancel();
    let result = if !uploads_complete {
        let uploaded = uploads
            .await
            .context("upload task panicked")
            .and_then(|result| result);
        result.and(uploaded)
    } else {
        result
    };
    let ended = end_run(grpc, initialized.run_id).await;
    result.and(ended)
}

async fn end_run(mut grpc: TensorLaneClient<Channel>, run_id: String) -> anyhow::Result<()> {
    tokio::time::timeout(Duration::from_secs(30), grpc.end(EndRequest { run_id }))
        .await
        .context("End RPC timed out")?
        .context("End RPC failed")?;
    Ok(())
}
