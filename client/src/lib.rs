use pyo3::prelude::*;

const MAX_BATCH_BYTES: usize = 64 * 1024 * 1024;

mod assets;
mod client;
mod data;
mod ipc;
mod semaphore;
mod uploads;
mod worker;

mod proto {
    tonic::include_proto!("_");
}

#[pymodule]
mod _native {
    #[pymodule_export]
    use crate::client::{Daemon, Listener, Semaphore};
    #[pymodule_export]
    use crate::uploads::UploadClient;
}
