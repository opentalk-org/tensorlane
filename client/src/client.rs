use std::sync::Mutex;

use anyhow::{Context, anyhow, ensure};
use pyo3::prelude::*;
use pyo3_tch::PyTensor;
use tch::Tensor;
use tokio::sync::oneshot;

use crate::proto::Sample;
use crate::worker::{Command, Worker};

type SampleParts = (PyTensor, f64, i64, i32, PyTensor);

fn sample_parts(sample: Sample) -> anyhow::Result<SampleParts> {
    ensure!(
        sample.wave.len() % 2 == 0,
        "wave byte length must be a multiple of 2 (int16 PCM)"
    );
    ensure!(
        sample.text.len() % 8 == 0,
        "text byte length must be a multiple of 8 (int64 token IDs)"
    );
    // Decode explicitly instead of assuming host endianness or byte alignment.
    let wave: Vec<i16> = sample
        .wave
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let text: Vec<i64> = sample
        .text
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    Ok((
        PyTensor(Tensor::f_from_slice(&wave)?),
        sample.duration,
        sample.speaker_id,
        sample.language_id,
        PyTensor(Tensor::f_from_slice(&text)?),
    ))
}

#[pyclass]
pub struct Client {
    worker: Mutex<Option<Worker>>,
    #[pyo3(get)]
    run_id: String,
    #[pyo3(get)]
    train_config: String,
}

impl Client {
    fn start(run_id: String, addr: String) -> anyhow::Result<Self> {
        let (worker, initialized) = Worker::start(run_id, addr)?;
        Ok(Self {
            worker: Mutex::new(Some(worker)),
            run_id: initialized.run_id,
            train_config: initialized.train_config,
        })
    }

    fn enqueue(&self, command: Command) -> anyhow::Result<()> {
        let guard = self
            .worker
            .lock()
            .map_err(|_| anyhow!("worker lock poisoned"))?;
        let worker = guard.as_ref().context("TensorLane client is closed")?;
        worker.send(command)
    }

    fn stop(&self) -> anyhow::Result<()> {
        let worker = self
            .worker
            .lock()
            .map_err(|_| anyhow!("worker lock poisoned"))?
            .take();
        if let Some(mut worker) = worker {
            worker.shutdown()?;
        }
        Ok(())
    }
}

#[pymethods]
impl Client {
    #[new]
    #[pyo3(signature = (run_id, addr="localhost:8181"))]
    fn new(py: Python<'_>, run_id: String, addr: &str) -> anyhow::Result<Self> {
        py.allow_threads(|| Self::start(run_id, addr.to_owned()))
    }

    #[pyo3(signature = (validation=false))]
    fn next_batch(
        &self,
        py: Python<'_>,
        validation: bool,
    ) -> anyhow::Result<Option<Vec<SampleParts>>> {
        py.allow_threads(|| {
            let (reply, response) = oneshot::channel();
            self.enqueue(Command::NextBatch { validation, reply })?;
            let batch = response
                .blocking_recv()
                .context("worker dropped the batch reply")??;
            batch
                .map(|response| response.batch.into_iter().map(sample_parts).collect())
                .transpose()
        })
    }

    fn close(&self, py: Python<'_>) -> anyhow::Result<()> {
        py.allow_threads(|| self.stop())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind};

    #[test]
    fn pytensor_exposes_a_torch_tensor_without_copying_its_storage() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            py.import("torch").unwrap();
            let tensor = Tensor::from_slice(&[1_i64, 2, 3]);
            let pointer = tensor.data_ptr() as usize;
            let object = PyTensor(tensor).into_pyobject(py).unwrap();
            assert_eq!(
                object
                    .call_method0("data_ptr")
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                pointer
            );
            assert_eq!(
                object
                    .call_method0("tolist")
                    .unwrap()
                    .extract::<Vec<i64>>()
                    .unwrap(),
                [1, 2, 3]
            );
        });
    }

    #[test]
    fn decodes_owned_tensors_with_wire_dtypes_and_values() {
        let waves = [i16::MIN, -1, 0, i16::MAX];
        let tokens = [0_i64, 123, i64::MAX];
        let (wave, duration, speaker, language, text) = sample_parts(Sample {
            wave: waves.into_iter().flat_map(i16::to_le_bytes).collect(),
            text: tokens.into_iter().flat_map(i64::to_le_bytes).collect(),
            duration: 0.5,
            speaker_id: 42,
            language_id: 7,
        })
        .unwrap();
        assert_eq!(wave.kind(), Kind::Int16);
        assert_eq!(text.kind(), Kind::Int64);
        assert_eq!(wave.device(), Device::Cpu);
        assert_eq!(wave.size(), [4]);
        assert_eq!(text.size(), [3]);
        assert_eq!(Vec::<i16>::try_from(&wave.0).unwrap(), waves);
        assert_eq!(Vec::<i64>::try_from(&text.0).unwrap(), tokens);
        assert_eq!((duration, speaker, language), (0.5, 42, 7));
    }

    #[test]
    fn handles_empty_buffers_and_rejects_truncated_values() {
        let (wave, _, _, _, text) = sample_parts(Sample::default()).unwrap();
        assert_eq!(wave.size(), [0]);
        assert_eq!(text.size(), [0]);
        assert!(
            sample_parts(Sample {
                wave: vec![0],
                ..Default::default()
            })
            .is_err()
        );
        assert!(
            sample_parts(Sample {
                text: vec![0; 7],
                ..Default::default()
            })
            .is_err()
        );
    }
}
