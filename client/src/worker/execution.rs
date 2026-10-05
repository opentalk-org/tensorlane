use super::{Initialized, Options, Settings};
use crate::{
    data::{Work, prefetch},
    ipc::Sender,
    semaphore::BatchBudget,
};
use anyhow::{Context, anyhow, ensure};
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use tokio::{
    net::UnixListener,
    sync::{mpsc, oneshot},
};
use tokio_util::sync::CancellationToken;

pub(super) async fn supervise(
    options: Options,
    session: String,
    budgets: Arc<Mutex<HashMap<String, Arc<BatchBudget>>>>,
    mut stop: oneshot::Receiver<anyhow::Result<()>>,
    ready: &std::sync::mpsc::SyncSender<anyhow::Result<Initialized>>,
    connected: &AtomicBool,
) -> anyhow::Result<()> {
    let work_listener = UnixListener::bind(options.root.join("work.sock"))?;
    let upload_listener = UnixListener::bind(options.root.join("uploads.sock"))?;
    let mut remote = None;
    let startup_timeout = options.startup_timeout;
    let startup = async {
        let http =
            crate::transport::connect(&options.addr, options.api_key.as_deref(), session).await?;
        let initialized = http.initialize(&options.run_id).await?;
        remote = Some((http.clone(), initialized.run_id.clone()));
        let heartbeat = tokio_util::task::AbortOnDropHandle::new(tokio::spawn(send_heartbeats(
            http.clone(),
            initialized.run_id.clone(),
        )));
        let settings = Settings::resolve(&options, &initialized.config)?;
        std::fs::write(options.root.join("ranks"), settings.ranks.to_string())?;
        ensure!(
            !initialized.streams.is_empty(),
            "server returned no streams"
        );
        ensure!(
            !initialized
                .streams
                .iter()
                .any(|name| matches!(name.as_str(), "." | "..")),
            "stream name must not be . or .."
        );
        let (assets, asset_metadata) =
            crate::assets::prefetch(&http, &initialized, &options.root).await?;
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
                    (settings.memory_bytes / initialized.streams.len()).max(1),
                )?);
                let directory = options.root.join("streams").join(index.to_string());
                std::fs::create_dir_all(&directory)?;
                std::fs::write(directory.join("semaphore"), budget.semaphore.name()?)?;
                std::fs::write(directory.join("memory"), budget.memory.semaphore.name()?)?;
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
        let mut work = Vec::new();
        for _ in 0..settings.num_workers {
            work.push(Sender::<Work, _>::new(work_listener.accept().await?.0));
        }
        anyhow::Ok((http, initialized, work, heartbeat))
    };
    let started = tokio::select! {
        result = startup => result.map(Some),
        result = &mut stop => result.unwrap_or(Ok(())).map(|()| None),
        _ = async {
            match startup_timeout {
                Some(timeout) => tokio::time::sleep(timeout).await,
                None => std::future::pending().await,
            }
        } => Err(anyhow!("TensorLane startup timed out")),
    };
    let (http, initialized, work, mut heartbeat) = match started {
        Ok(Some(started)) => started,
        result => {
            let ended = match remote {
                Some((http, run_id)) => http
                    .end(
                        &run_id,
                        result.is_err() || options.root.join("failed").exists(),
                    )
                    .await
                    .context("ending run failed"),
                None => Ok(()),
            };
            return result.map(|_| ()).and(ended);
        }
    };
    let uploads_stopping = CancellationToken::new();
    let mut uploads = tokio::spawn(crate::uploads::serve(
        upload_listener,
        http.clone(),
        initialized.run_id.clone(),
        uploads_stopping.clone(),
    ));
    let mut uploads_complete = false;
    let (send_work, mut receive_work) = mpsc::unbounded_channel::<Work>();
    let mut sender = tokio::spawn(async move {
        let mut senders = work;
        let mut next_worker = 0;
        while let Some(message) = receive_work.recv().await {
            match message {
                Work::End { ref stream } => {
                    for sender in &mut senders {
                        sender
                            .send(&Work::End {
                                stream: stream.clone(),
                            })
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
                    futures_util::future::try_join_all(senders.iter_mut().zip(parts).map(
                        |(sender, samples)| {
                            let stream = stream.clone();
                            async move {
                                if !samples.is_empty() {
                                    sender
                                        .send(&Work::Batch {
                                            stream,
                                            batch,
                                            query_batch_idx,
                                            timings,
                                            samples,
                                            memory_units,
                                        })
                                        .await
                                        .context("transform worker disconnected")?;
                                }
                                anyhow::Ok(())
                            }
                        },
                    ))
                    .await?;
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
            http.clone(),
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
                biased;
                result = &mut stop => return result.unwrap_or(Ok(())),
                result = &mut heartbeat => {
                    result.context("heartbeat task panicked")??;
                    return Err(anyhow!("heartbeat stopped unexpectedly"));
                },
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
    let failed = result.is_err() || options.root.join("failed").exists();
    let ended = http
        .end(&initialized.run_id, failed)
        .await
        .context("ending run failed");
    result.and(ended)
}

async fn send_heartbeats(http: crate::transport::HttpClient, run_id: String) -> anyhow::Result<()> {
    loop {
        if let Err(error) = http.heartbeat(&run_id).await {
            eprintln!("TensorLane heartbeat failed: {error:#}");
        }
        tokio::time::sleep(Duration::from_secs(10)).await;
    }
}
