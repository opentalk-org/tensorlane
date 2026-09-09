use pyo3::prelude::*;

mod client;
mod data;
mod ipc;
mod semaphore;
mod worker;

mod proto {
    tonic::include_proto!("_");
}

#[pymodule]
mod _native {
    #[pymodule_export]
    use crate::client::{Daemon, Listener, Semaphore};
}
