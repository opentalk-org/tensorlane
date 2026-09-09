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
use std::{os::unix::net::UnixStream, path::PathBuf, sync::Mutex};

#[pyclass]
pub struct Daemon {
    worker: Mutex<Option<Worker>>,
    #[pyo3(get)]
    run_id: String,
    #[pyo3(get)]
    train_config: String,
}
#[pymethods]
impl Daemon {
    #[new]
    fn new(
        py: Python<'_>,
        run_id: String,
        addr: String,
        root: PathBuf,
        ranks: usize,
        prefetch_factor: usize,
        num_workers: usize,
    ) -> anyhow::Result<Self> {
        py.allow_threads(|| {
            let (worker, initialized) = Worker::start(Options {
                run_id,
                addr,
                root,
                ranks,
                factor: prefetch_factor,
                num_workers,
            })?;
            Ok(Self {
                worker: Mutex::new(Some(worker)),
                run_id: initialized.run_id,
                train_config: initialized.train_config,
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
    socket: UnixStream,
    runtime: tokio::runtime::Runtime,
}
#[pymethods]
impl Listener {
    #[new]
    fn new(py: Python<'_>, socket: &Bound<'_, PyAny>) -> anyhow::Result<Self> {
        let duplicate = socket.call_method0("dup")?;
        let descriptor = duplicate
            .call_method0("detach")?
            .extract::<std::os::fd::RawFd>()?;
        use std::os::fd::FromRawFd;
        let stream = unsafe { UnixStream::from_raw_fd(descriptor) };
        py.allow_threads(|| {
            stream.set_nonblocking(true)?;
            let socket = stream.try_clone()?;
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()?;
            let stream = {
                let _entered = runtime.enter();
                tokio::net::UnixStream::from_std(stream)?
            };
            let (messages, receiver) = tokio::sync::mpsc::unbounded_channel();
            runtime.spawn(async move {
                let mut reader = Receiver::<Work, _>::new(stream);
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
            Ok(Self {
                receiver: Mutex::new(receiver),
                socket,
                runtime,
            })
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
                batch,
                index,
                wave,
                text,
                duration,
                speaker_id,
                language_id,
            } => {
                object.set_item("kind", "sample")?;
                object.set_item("batch", batch)?;
                object.set_item("index", index)?;
                object.set_item("wave", PyBytes::new(py, &wave))?;
                object.set_item("text", PyBytes::new(py, &text))?;
                object.set_item("duration", duration)?;
                object.set_item("speaker_id", speaker_id)?;
                object.set_item("language_id", language_id)?;
            }
            Work::End => {
                object.set_item("kind", "end")?;
            }
        }
        Ok(Some(object.into_any().unbind()))
    }

    fn close(&self) {
        let _ = self.socket.shutdown(std::net::Shutdown::Both);
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
