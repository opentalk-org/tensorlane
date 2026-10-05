use crate::{
    data::Work,
    ipc::Receiver,
    semaphore::PosixSemaphore,
    worker::{Options, Worker},
};
use anyhow::anyhow;
use pyo3::{
    prelude::*,
    types::{PyBytes, PyDict},
};
use std::{collections::HashMap, path::PathBuf, sync::Mutex};

#[pyclass]
pub struct Daemon {
    worker: Mutex<Option<Worker>>,
    #[pyo3(get)]
    run_id: String,
    #[pyo3(get)]
    config: String,
    #[pyo3(get)]
    assets: HashMap<String, PathBuf>,
    #[pyo3(get)]
    streams: Vec<String>,
    #[pyo3(get)]
    asset_metadata: HashMap<String, String>,
    #[pyo3(get)]
    ranks: usize,
    #[pyo3(get)]
    num_workers: usize,
    #[pyo3(get)]
    prefetch_factor: usize,
}
#[pymethods]
impl Daemon {
    #[new]
    #[pyo3(signature=(run_id, addr, root, ranks=None, prefetch_factor=None, num_workers=None, rank=0, api_key=None, timeout=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        run_id: String,
        addr: String,
        root: PathBuf,
        ranks: Option<usize>,
        prefetch_factor: Option<usize>,
        num_workers: Option<usize>,
        rank: usize,
        api_key: Option<String>,
        timeout: Option<f64>,
    ) -> anyhow::Result<Self> {
        py.allow_threads(|| {
            let (worker, initialized) = Worker::start(Options {
                run_id,
                addr,
                api_key,
                root,
                ranks,
                factor: prefetch_factor,
                num_workers,
                rank,
                startup_timeout: timeout
                    .map(std::time::Duration::try_from_secs_f64)
                    .transpose()?,
            })?;
            Ok(Self {
                worker: Mutex::new(Some(worker)),
                run_id: initialized.response.run_id,
                config: initialized.response.config,
                assets: initialized.assets,
                streams: initialized.response.streams,
                asset_metadata: initialized.asset_metadata,
                ranks: initialized.settings.ranks,
                num_workers: initialized.settings.num_workers,
                prefetch_factor: initialized.settings.factor,
            })
        })
    }
    #[pyo3(signature = (error=None))]
    fn stop(&self, error: Option<String>) -> anyhow::Result<()> {
        if let Some(worker) = self
            .worker
            .lock()
            .map_err(|_| anyhow!("daemon lock poisoned"))?
            .as_mut()
        {
            worker.stop(error);
        }
        Ok(())
    }
    fn check(&self) -> anyhow::Result<()> {
        self.worker
            .lock()
            .map_err(|_| anyhow!("daemon lock poisoned"))?
            .as_ref()
            .ok_or_else(|| anyhow!("TensorLane daemon is closed"))?
            .check()
    }
    fn ready(&self) -> anyhow::Result<bool> {
        self.worker
            .lock()
            .map_err(|_| anyhow!("daemon lock poisoned"))?
            .as_ref()
            .ok_or_else(|| anyhow!("TensorLane daemon is closed"))?
            .ready()
    }
    fn close(&self, py: Python<'_>) -> anyhow::Result<()> {
        py.allow_threads(|| {
            if let Some(mut worker) = self
                .worker
                .lock()
                .map_err(|_| anyhow!("daemon lock poisoned"))?
                .take()
            {
                worker.shutdown()?;
            }
            Ok(())
        })
    }
}
#[pyclass]
pub struct Listener {
    receiver: Mutex<tokio::sync::mpsc::UnboundedReceiver<anyhow::Result<Work>>>,
    runtime: tokio::runtime::Runtime,
}
#[pymethods]
impl Listener {
    #[new]
    fn new(py: Python<'_>, path: PathBuf) -> anyhow::Result<Self> {
        py.allow_threads(|| {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            let receiver = runtime.block_on(async move {
                let stream = tokio::net::UnixStream::connect(path).await?;
                let mut reader = Receiver::<Work, _>::new(stream);

                let (messages, receiver) = tokio::sync::mpsc::unbounded_channel();
                tokio::spawn(async move {
                    loop {
                        match reader.recv().await {
                            Ok(Some(message)) => {
                                if messages.send(Ok(message)).is_err() {
                                    break;
                                }
                            }
                            Ok(None) => break,
                            Err(error) => {
                                let _ = messages.send(Err(error));
                                break;
                            }
                        }
                    }
                });

                anyhow::Ok(Mutex::new(receiver))
            })?;
            anyhow::Ok(Self { receiver, runtime })
        })
    }
    fn recv(&self, py: Python<'_>) -> anyhow::Result<Option<Py<PyAny>>> {
        let message = py.allow_threads(|| {
            let mut receiver = self
                .receiver
                .lock()
                .map_err(|_| anyhow!("listener lock poisoned"))?;
            self.runtime.block_on(receiver.recv()).transpose()
        })?;
        let Some(message) = message else {
            return Ok(None);
        };
        let object = PyDict::new(py);
        match message {
            Work::Sample {
                stream,
                batch,
                query_batch_idx,
                timings,
                index,
                sample,
            } => {
                object.set_item("kind", "sample")?;
                object.set_item("stream", stream)?;
                object.set_item("batch", batch)?;
                object.set_item("query_batch_idx", query_batch_idx)?;
                object.set_item("timings", timings)?;
                object.set_item("index", index)?;
                object.set_item("sample_id", sample.sample_id)?;
                object.set_item("metadata_json", sample.metadata_json)?;
                let values = PyDict::new(py);
                for (name, value) in sample.blobs {
                    values.set_item(name, PyBytes::new(py, &value))?;
                }
                object.set_item("blobs", values)?;
            }
            Work::End { stream } => {
                object.set_item("kind", "end")?;
                object.set_item("stream", stream)?;
            }
        }
        Ok(Some(object.into_any().unbind()))
    }
}

#[pyclass]
pub struct Semaphore {
    inner: PosixSemaphore,
}

#[pymethods]
impl Semaphore {
    #[new]
    fn new(name: &str) -> anyhow::Result<Self> {
        Ok(Self {
            inner: PosixSemaphore::open(name)?,
        })
    }

    fn post(&self, py: Python<'_>) -> anyhow::Result<()> {
        py.allow_threads(|| self.inner.post())
    }
}
