use std::sync::Mutex;

use anyhow::{Context, ensure};
use flume::Receiver;
use pyo3::prelude::*;
use pyo3_tch::PyTensor;
use tch::{Device, Kind, Tensor};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::audio::{AudioProcessor, ProcessedSample};
use crate::data_pipeline::Pipeline;
use crate::proto::DataResponse;
use crate::proto::give_me_data_client::GiveMeDataClient;

type BatchParts = (
    Vec<PyTensor>,
    Vec<f64>,
    PyTensor,
    PyTensor,
    PyTensor,
    PyTensor,
    PyTensor,
    PyTensor,
    PyTensor,
);

pub struct NativeBatch {
    waves: Vec<Tensor>,
    durations: Vec<f64>,
    speaker_ids: Tensor,
    language_ids: Tensor,
    modality_ids: Tensor,
    texts: Tensor,
    input_lengths: Tensor,
    mels: Tensor,
    mel_lengths: Tensor,
}

impl NativeBatch {
    fn into_python(self) -> BatchParts {
        (
            self.waves.into_iter().map(PyTensor).collect(),
            self.durations,
            PyTensor(self.speaker_ids),
            PyTensor(self.language_ids),
            PyTensor(self.modality_ids),
            PyTensor(self.texts),
            PyTensor(self.input_lengths),
            PyTensor(self.mels),
            PyTensor(self.mel_lengths),
        )
    }
}

#[pyclass(name = "DataStream")]
pub struct NativeDataStream {
    receiver: Mutex<Receiver<anyhow::Result<NativeBatch>>>,
    cancellation: CancellationToken,
}

#[pymethods]
impl NativeDataStream {
    fn __iter__(this: PyRef<'_, Self>) -> PyRef<'_, Self> {
        this
    }

    fn __next__(&self, py: Python<'_>) -> anyhow::Result<Option<BatchParts>> {
        let receiver = self
            .receiver
            .lock()
            .map_err(|_| anyhow::anyhow!("data stream lock is poisoned"))?;
        let result = py.allow_threads(|| receiver.recv());
        match result {
            Ok(Ok(batch)) => Ok(Some(batch.into_python())),
            Ok(Err(error)) => Err(error),
            Err(_) => Ok(None),
        }
    }
}

impl Drop for NativeDataStream {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

pub struct DataTask {
    pub cancellation: CancellationToken,
    pub join: JoinHandle<anyhow::Result<()>>,
}

pub fn spawn(
    runtime: &tokio::runtime::Runtime,
    mut client: GiveMeDataClient<tonic::transport::Channel>,
    run_id: String,
    validation: bool,
    prefetch: usize,
    modality_id: i64,
    pin_memory: bool,
) -> anyhow::Result<(NativeDataStream, DataTask)> {
    ensure!(prefetch > 0, "prefetch must be greater than zero");
    let (output_tx, output_rx) = flume::bounded(prefetch);
    let cancellation = CancellationToken::new();
    let task_cancellation = cancellation.clone();
    let pipeline = Pipeline {
        run_id,
        validation,
        prefetch,
        modality_id,
        pin_memory,
        output: output_tx,
        cancellation: task_cancellation,
    };
    let join = runtime.spawn(async move { pipeline.run(&mut client).await });
    Ok((
        NativeDataStream {
            receiver: Mutex::new(output_rx),
            cancellation: cancellation.clone(),
        },
        DataTask { cancellation, join },
    ))
}

pub(crate) fn process_response(
    processor: &AudioProcessor,
    response: DataResponse,
    modality_id: i64,
    pin_memory: bool,
) -> anyhow::Result<NativeBatch> {
    ensure!(!response.batch.is_empty(), "server returned an empty batch");
    let samples = response
        .batch
        .into_iter()
        .map(|sample| {
            processor.process(
                &sample.wave,
                &sample.text,
                sample.duration,
                sample.speaker_id,
                sample.language_id,
            )
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut samples = samples
        .into_iter()
        .map(|sample| anyhow::Ok((tensor_dimension(&sample.mel, 1, "mel")?, sample)))
        .collect::<anyhow::Result<Vec<_>>>()?;
    samples.sort_by_key(|(length, _)| std::cmp::Reverse(*length));
    let samples = samples.into_iter().map(|(_, sample)| sample).collect();
    collate(samples, modality_id, pin_memory)
}

fn collate(
    samples: Vec<ProcessedSample>,
    modality_id: i64,
    pin_memory: bool,
) -> anyhow::Result<NativeBatch> {
    let batch_size = samples.len() as i64;
    let max_mel = samples
        .iter()
        .map(|sample| tensor_dimension(&sample.mel, 1, "mel"))
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .max()
        .context("missing mel")?;
    let max_text = samples
        .iter()
        .map(|sample| tensor_dimension(&sample.text, 0, "text"))
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .max()
        .context("missing text")?;
    let mut speaker_ids = Vec::with_capacity(samples.len());
    let mut language_ids = Vec::with_capacity(samples.len());
    let mut input_lengths = Vec::with_capacity(samples.len());
    let mut mel_lengths = Vec::with_capacity(samples.len());
    let mut waves = Vec::with_capacity(samples.len());
    let mut durations = Vec::with_capacity(samples.len());
    let texts = Tensor::f_zeros([batch_size, max_text], (Kind::Int64, Device::Cpu))?;
    let mels = Tensor::f_zeros([batch_size, 80, max_mel], (Kind::Float, Device::Cpu))?;
    for (index, sample) in samples.into_iter().enumerate() {
        let text_length = tensor_dimension(&sample.text, 0, "text")?;
        let mel_length = tensor_dimension(&sample.mel, 1, "mel")?;
        let mut text_target = texts.f_get(index as i64)?.f_narrow(0, 0, text_length)?;
        text_target.f_copy_(&sample.text)?;
        let mut mel_target = mels.f_get(index as i64)?.f_narrow(1, 0, mel_length)?;
        mel_target.f_copy_(&sample.mel)?;
        speaker_ids.push(sample.speaker_id);
        language_ids.push(sample.language_id);
        input_lengths.push(text_length);
        mel_lengths.push(mel_length);
        waves.push(maybe_pin(sample.wave, pin_memory)?);
        durations.push(sample.duration);
    }
    Ok(NativeBatch {
        waves,
        durations,
        speaker_ids: maybe_pin(Tensor::f_from_slice(&speaker_ids)?, pin_memory)?,
        language_ids: maybe_pin(Tensor::f_from_slice(&language_ids)?, pin_memory)?,
        modality_ids: maybe_pin(
            Tensor::f_full([batch_size], modality_id, (Kind::Int64, Device::Cpu))?,
            pin_memory,
        )?,
        texts: maybe_pin(texts, pin_memory)?,
        input_lengths: maybe_pin(Tensor::f_from_slice(&input_lengths)?, pin_memory)?,
        mels: maybe_pin(mels, pin_memory)?,
        mel_lengths: maybe_pin(Tensor::f_from_slice(&mel_lengths)?, pin_memory)?,
    })
}

fn maybe_pin(tensor: Tensor, pin: bool) -> anyhow::Result<Tensor> {
    if pin {
        Ok(tensor.f_pin_memory(Device::Cuda(0))?)
    } else {
        Ok(tensor)
    }
}

fn tensor_dimension(tensor: &Tensor, dimension: usize, name: &str) -> anyhow::Result<i64> {
    tensor
        .size()
        .get(dimension)
        .copied()
        .with_context(|| format!("{name} tensor has no dimension {dimension}"))
}
