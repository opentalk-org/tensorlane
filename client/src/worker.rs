use crate::{
    data::{Work, prefetch},
    ipc::Sender,
    proto::{InitRequest, InitResponse, tensor_lane_client::TensorLaneClient},
    semaphore::BatchBudget,
};
use anyhow::{Context, anyhow, ensure};
use fs2::FileExt;
use std::{
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
use tonic::transport::Endpoint;

pub struct Options {
    pub run_id: String,
    pub addr: String,
    pub root: PathBuf,
    pub ranks: usize,
    pub factor: usize,
    pub num_workers: usize,
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
    budgets: [Arc<BatchBudget>; 2],
    _lock: File,
}
impl Resources {
    fn new(root: &Path, capacity: usize, ranks: usize) -> anyhow::Result<Self> {
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
                std::fs::remove_file(entry.path())?;
            }
        }
        let mut auth = File::create(root.join("auth"))?;
        auth.write_all(uuid::Uuid::new_v4().as_bytes())?;
        auth.write_all(uuid::Uuid::new_v4().as_bytes())?;
        let budgets = [
            Arc::new(BatchBudget::new(capacity)?),
            Arc::new(BatchBudget::new(capacity)?),
        ];
        for (name, budget) in ["semaphore", "validation-semaphore"]
            .into_iter()
            .zip(&budgets)
        {
            std::fs::write(root.join(name), budget.semaphore.name()?)?;
        }
        std::fs::write(root.join("ranks"), ranks.to_string())?;
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
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
    }
}
impl Worker {
    pub fn start(options: Options) -> anyhow::Result<(Self, InitResponse)> {
        ensure!(
            options.ranks > 0 && options.factor > 0,
            "ranks and prefetch_factor must be positive"
        );
        ensure!(options.num_workers > 0, "num_workers must be positive");
        let capacity = options
            .ranks
            .checked_mul(options.factor)
            .context("prefetch capacity overflow")?;
        let resources = Resources::new(&options.root, capacity, options.ranks)?;
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
        for budget in &self._resources.budgets {
            budget.cancel();
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
    budgets: [Arc<BatchBudget>; 2],
    mut stop: oneshot::Receiver<anyhow::Result<()>>,
    ready: &std::sync::mpsc::SyncSender<anyhow::Result<InitResponse>>,
    connected: &AtomicBool,
) -> anyhow::Result<()> {
    let work_listener = UnixListener::bind(options.root.join("work.sock"))?;
    let url = if options.addr.contains("://") {
        options.addr.clone()
    } else {
        format!("http://{}", options.addr)
    };
    let startup = async {
        let channel = Endpoint::from_shared(url)?
            .connect_timeout(Duration::from_secs(10))
            .connect()
            .await?;
        let mut grpc = TensorLaneClient::new(channel).max_decoding_message_size(67_136_000);
        let initialized = grpc
            .init(InitRequest {
                run_id: options.run_id.clone(),
            })
            .await?
            .into_inner();
        ready
            .send(Ok(initialized.clone()))
            .map_err(|_| anyhow!("initializer disconnected"))?;
        let mut sockets = Vec::new();
        for _ in 0..options.num_workers {
            sockets.push(work_listener.accept().await?.0);
        }
        let mut work = Vec::new();
        for socket in sockets {
            work.push(Sender::<Work, _>::new(socket));
        }
        anyhow::Ok((grpc, initialized, work))
    };
    let (grpc, initialized, work) = tokio::select! {
        result = tokio::time::timeout(Duration::from_secs(120), startup) => result.context("daemon startup timed out")??,
        result = &mut stop => return result.unwrap_or(Ok(())),
    };
    let (send_work, mut receive_work) = mpsc::unbounded_channel::<Work>();
    let mut sender = tokio::spawn(async move {
        let mut senders = work;
        let mut next_worker = 0;
        while let Some(message) = receive_work.recv().await {
            match &message {
                Work::End { .. } => {
                    for sender in &mut senders {
                        sender.send(&message).await?;
                    }
                }
                Work::Sample { .. } => {
                    senders[next_worker].send(&message).await?;
                    next_worker = (next_worker + 1) % senders.len();
                }
            }
        }
        anyhow::Ok(senders)
    });
    let mut pumps = tokio::task::JoinSet::new();
    for (validation, budget) in [false, true].into_iter().zip(&budgets) {
        pumps.spawn(prefetch(
            grpc.clone(),
            initialized.run_id.clone(),
            validation,
            budget.clone(),
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
    for budget in &budgets {
        budget.cancel();
    }
    pumps.abort_all();
    while pumps.join_next().await.is_some() {}
    drop(idle_senders);
    if !sender_complete {
        sender.abort();
        let _ = sender.await;
    }
    result
}
