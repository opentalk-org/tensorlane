use super::{Initialized, Options, Settings};
use crate::{
    assets::{Lease, Prefetched},
    data::{Work, prefetch},
    ipc::Sender,
    semaphore::BatchBudget,
    transport::HttpClient,
};
use anyhow::{Context, anyhow, ensure};
use reqwest::Method;
use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::SyncSender,
    },
    time::Duration,
};
use tokio::{
    net::UnixListener,
    sync::{mpsc, oneshot},
    task::JoinSet,
    time::MissedTickBehavior,
};
use tokio_util::{sync::CancellationToken, task::AbortOnDropHandle};

type Budgets = Vec<(String, Arc<BatchBudget>, u64)>;

pub(super) async fn supervise(
    options: Options,
    mut stop: oneshot::Receiver<anyhow::Result<()>>,
    ready: SyncSender<Initialized>,
    connected: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let work_listener = UnixListener::bind(options.root.join("work.sock"))?;
    let upload_listener = UnixListener::bind(options.root.join("uploads.sock"))?;
    let mut remote = None;
    let started = tokio::select! {
        result = start(&options, &work_listener, ready, &mut remote) => result.map(Some),
        result = &mut stop => result.unwrap_or(Ok(())).map(|()| None),
        _ = tokio::time::sleep(options.startup_timeout.unwrap_or(Duration::MAX)) => {
            Err(anyhow!("TensorLane startup timed out"))
        }
    };
    let Some((http, run_id)) = remote else {
        return started.map(|_| ());
    };
    let (work, budgets, _leases, heartbeat) = match started {
        Ok(Some(started)) => started,
        result => {
            let failed = result.is_err() || options.root.join("failed").exists();
            let ended = http.end(&run_id, failed).await.context("ending run failed");
            return result.map(|_| ()).and(ended);
        }
    };
    let uploads_stopping = CancellationToken::new();
    let mut uploads = tokio::spawn(crate::uploads::serve(
        upload_listener,
        http.clone(),
        run_id.clone(),
        uploads_stopping.clone(),
    ));
    let (send_work, receive_work) = mpsc::unbounded_channel();
    let mut sender = AbortOnDropHandle::new(tokio::spawn(distribute(work, receive_work)));
    let mut pumps = JoinSet::new();
    for (name, budget, start) in &budgets {
        pumps.spawn(prefetch(
            http.clone(),
            run_id.clone(),
            name.clone(),
            budget.clone(),
            send_work.clone(),
            *start,
        ));
    }
    drop(send_work);
    connected.store(true, Ordering::Release);
    let mut uploads_complete = false;
    let result = async {
        loop {
            tokio::select! {
                biased;
                result = &mut stop => return result.unwrap_or(Ok(())),
                result = &mut uploads => {
                    uploads_complete = true;
                    result.context("upload task panicked")??;
                    return Err(anyhow!("upload listener stopped unexpectedly"));
                }
                Some(result) = pumps.join_next() => result.context("prefetch task panicked")??,
                result = &mut sender => return result.context("socket sender panicked")?,
            }
        }
    }
    .await;
    let _ = std::fs::remove_file(options.root.join("init.json"));
    for (_, budget, _) in &budgets {
        budget.cancel();
    }
    pumps.abort_all();
    while pumps.join_next().await.is_some() {}
    drop(sender);
    uploads_stopping.cancel();
    let result = if uploads_complete {
        result
    } else {
        let uploaded = uploads.await.context("upload task panicked");
        result.and(uploaded.and_then(|uploaded| uploaded))
    };
    drop(heartbeat);
    let failed = result.is_err() || options.root.join("failed").exists();
    let ended = http.end(&run_id, failed).await.context("ending run failed");
    result.and(ended)
}

async fn start(
    options: &Options,
    work_listener: &UnixListener,
    ready: SyncSender<Initialized>,
    remote: &mut Option<(HttpClient, String)>,
) -> anyhow::Result<(
    Vec<Sender<Work>>,
    Budgets,
    Vec<Lease>,
    AbortOnDropHandle<()>,
)> {
    let http = crate::transport::connect(&options.addr, options.api_key.as_deref()).await?;
    let initialized = http.initialize(&options.run_id).await?;
    *remote = Some((http.clone(), initialized.run_id.clone()));
    let heartbeat = AbortOnDropHandle::new(tokio::spawn(heartbeat(
        http.clone(),
        initialized.run_id.clone(),
    )));
    let settings = Settings::resolve(options, &initialized.config)?;
    std::fs::write(options.root.join("ranks"), settings.ranks.to_string())?;
    ensure!(
        !initialized.streams.is_empty(),
        "server returned no streams"
    );
    let Prefetched {
        paths,
        metadata,
        leases,
    } = crate::assets::prefetch(&http, &initialized, settings.memory_bytes).await?;
    let mut next_batches = HashMap::new();
    if let Some(checkpoint) = &initialized.checkpoint {
        let mut checkpoint: serde_json::Value = serde_json::from_str(&metadata[&checkpoint.name])?;
        let next = checkpoint["metadata"]["_tensorlane"]["next_batches"].take();
        next_batches =
            serde_json::from_value::<Option<HashMap<String, u64>>>(next)?.unwrap_or_default();
    }
    let ranks = settings.ranks as u64;
    let mut budgets = Vec::new();
    for (index, name) in initialized.streams.iter().enumerate() {
        let budget = Arc::new(BatchBudget::new(
            settings.ranks * settings.factor,
            settings.memory_bytes / initialized.streams.len(),
        )?);
        let directory = options.root.join("streams").join(index.to_string());
        std::fs::create_dir_all(&directory)?;
        let start = next_batches.get(name).copied().unwrap_or(0);
        std::fs::write(directory.join("start"), start.to_string())?;
        for rank in 0..ranks {
            let next = start + (rank + ranks - start % ranks) % ranks;
            std::fs::write(directory.join(format!("progress-{rank}")), next.to_string())?;
        }
        std::fs::write(directory.join("semaphore"), budget.semaphore.name()?)?;
        std::fs::write(directory.join("memory"), budget.memory.semaphore.name()?)?;
        budgets.push((name.clone(), budget, start));
    }
    ready
        .send(Initialized {
            response: initialized,
            assets: paths,
            asset_metadata: metadata,
            settings,
        })
        .map_err(|_| anyhow!("initializer disconnected"))?;
    let mut work = Vec::new();
    for _ in 0..settings.num_workers {
        work.push(Sender::new(work_listener.accept().await?.0));
    }
    Ok((work, budgets, leases, heartbeat))
}

async fn heartbeat(http: HttpClient, run_id: String) {
    let mut tick = tokio::time::interval(Duration::from_secs(30));
    tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tick.tick().await;
        let path = ["runs", &run_id, "heartbeat"];
        if let Err(error) = http.request(Method::POST, &path, None, &[]).await {
            eprintln!("TensorLane heartbeat failed: {error:#}");
        }
    }
}

async fn distribute(
    mut senders: Vec<Sender<Work>>,
    mut work: mpsc::UnboundedReceiver<Work>,
) -> anyhow::Result<()> {
    let mut next_worker = 0;
    while let Some(message) = work.recv().await {
        match message {
            Work::End { .. } => {
                for sender in &mut senders {
                    sender
                        .send(&message)
                        .await
                        .context("transform worker disconnected")?;
                }
            }
            Work::Batch {
                stream,
                batch,
                query_batch_idx,
                timings,
                samples,
                memory_units,
            } => {
                let mut parts = vec![Vec::new(); senders.len()];
                for sample in samples {
                    parts[next_worker].push(sample);
                    next_worker = (next_worker + 1) % senders.len();
                }
                let mut sends = Vec::new();
                for (sender, samples) in senders.iter_mut().zip(parts) {
                    if samples.is_empty() {
                        continue;
                    }
                    let message = Work::Batch {
                        stream: stream.clone(),
                        batch,
                        query_batch_idx,
                        timings,
                        samples,
                        memory_units,
                    };
                    sends.push(async move { sender.send(&message).await });
                }
                futures_util::future::try_join_all(sends)
                    .await
                    .context("transform worker disconnected")?;
            }
        }
    }
    std::future::pending().await
}
