use pyo3::prelude::*;

use tensorlane_protocol::MAX_BATCH_BYTES;

mod assets;
mod client;
mod data;
mod ipc;
mod semaphore;
mod transport;
mod upload_client;
mod uploads;
mod worker;

#[pymodule]
mod _native {
    #[pymodule_export]
    use crate::client::{Daemon, Listener, Semaphore};
    #[pymodule_export]
    use crate::upload_client::UploadClient;
}
