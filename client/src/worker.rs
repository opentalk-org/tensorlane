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
    sync::Arc,
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
    _resources: Resources,
}

pub fn status(root: &Path, state: &str, error: Option<&str>) -> anyhow::Result<()> {
    let temporary = root.join("status.tmp");
    std::fs::write(
        &temporary,
        serde_json::to_vec(&serde_json::json!({"state": state, "error": error}))?,
    )?;
    std::fs::rename(temporary, root.join("status.json"))?;
    Ok(())
}
struct Resources {
    root: PathBuf,
    budget: Arc<BatchBudget>,
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
        let budget = Arc::new(BatchBudget::new(capacity)?);
        std::fs::write(root.join("semaphore"), budget.semaphore.name()?)?;
        std::fs::write(root.join("ranks"), ranks.to_string())?;
        status(root, "starting", None)?;
        Ok(Self {
            root: root.to_owned(),
            budget,
            _lock: lock,
        })
    }
}
impl Drop for Resources {
    fn drop(&mut self) {
        if let Ok(entries) = std::fs::read_dir(&self.root) {
            for entry in entries.flatten() {
                if entry.file_name() != "lock" && entry.file_name() != "status.json" {
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
        let budget = resources.budget.clone();
        let root = options.root.clone();
        let (stop, stopped) = oneshot::channel();
        let (ready, initialized) = std::sync::mpsc::sync_channel(1);
        let thread = thread::Builder::new()
            .name("tensorlane-daemon".into())
            .spawn(move || {
                let result = (|| -> anyhow::Result<()> {
                    let runtime = tokio::runtime::Builder::new_multi_thread()
                        .worker_threads(2)
                        .enable_all()
                        .build()?;
                    runtime.block_on(supervise(options, budget, stopped, &ready))
                })();
                if let Err(error) = result {
                    let message = format!("{error:#}");
                    let _ = status(&root, "failed", Some(&message));
                    let _ = ready.send(Err(anyhow!(message)));
                } else {
                    let _ = status(&root, "closed", None);
                }
            })?;
        let mut worker = Self {
            stop: Some(stop),
            thread: Some(thread),
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
        self._resources.budget.cancel();
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(error.map_or(Ok(()), |message| Err(anyhow!(message))));
        }
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
    budget: Arc<BatchBudget>,
    mut stop: oneshot::Receiver<anyhow::Result<()>>,
    ready: &std::sync::mpsc::SyncSender<anyhow::Result<InitResponse>>,
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
                Work::End => {
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
    let mut pump = tokio::spawn(prefetch(
        grpc,
        initialized.run_id.clone(),
        budget.clone(),
        send_work,
    ));
    let mut idle_senders = None;
    let mut complete = false;
    let mut sender_complete = false;
    let result = async {
        status(&options.root, "ready", None)?;
        loop {
            tokio::select! {
                result = &mut stop => return result.unwrap_or(Ok(())),
                result = &mut pump, if !complete => {
                    complete = true;
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
    let saved_status = match &result {
        Ok(()) => status(&options.root, "stopping", None),
        Err(error) => status(&options.root, "failed", Some(&format!("{error:#}"))),
    };
    budget.cancel();
    pump.abort();
    if !complete {
        let _ = pump.await;
    }
    drop(idle_senders);
    if !sender_complete {
        sender.abort();
        let _ = sender.await;
    }
    result.and(saved_status)
}
