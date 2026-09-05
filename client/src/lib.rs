use pyo3::prelude::*;

mod assets;
mod audio;
mod checkpoints;
mod client;
mod data;
mod data_pipeline;
mod metrics;

mod proto {
    tonic::include_proto!("_");
}

#[pymodule]
mod _native {
    #[pymodule_export]
    use crate::client::NativeClient;
    #[pymodule_export]
    use crate::data::NativeDataStream;
    #[pymodule_export]
    use crate::metrics::NativeMetrics;
}
