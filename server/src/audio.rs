use std::io::Cursor;

use bytes::{BufMut, Bytes, BytesMut};
use hound::{SampleFormat, WavReader, WavSpec};
use rubato::{
    Fft, FixedSync, Resampler,
    audioadapter_buffers::{SizeError, direct::InterleavedSlice},
};
use thiserror::Error;
use tracing::trace;

#[derive(Error, Debug)]
pub enum AudioError {
    #[error("wav encoding/decoding error: {0}")]
    Wav(#[from] hound::Error),
    #[error("size error: {0}")]
    SizeError(#[from] SizeError),
    #[error("resampler error: {0}")]
    ResamplerError(#[from] rubato::ResamplerConstructionError),
    #[error("resampling error: {0}")]
    ResamplingError(#[from] rubato::ResampleError),
    #[error("unsupported audio spec: {0:?}")]
    Unsupported(WavSpec),
}

fn quantize(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * 15f32.exp2()).round() as i16
}

fn resample_and_quantize(
    samples: Vec<f32>,
    sample_rate: u32,
    target_sample_rate: u32,
) -> Result<Vec<i16>, AudioError> {
    trace!(
        from = sample_rate,
        to = target_sample_rate,
        "resampling and quantizing audio"
    );
    let mut rs = Fft::<f32>::new(
        sample_rate as usize,
        target_sample_rate as usize,
        1024,
        1,
        FixedSync::Both,
    )?;

    let adapter = InterleavedSlice::new(&samples, 1, samples.len())?;
    let resampled = rs.process_all(&adapter, samples.len(), None)?;

    Ok(resampled.take_data().into_iter().map(quantize).collect())
}

pub fn process_audio(raw_wav: Bytes, target_sample_rate: u32) -> Result<Bytes, AudioError> {
    let mut reader = WavReader::new(Cursor::new(raw_wav))?;
    let spec = reader.spec();

    let samples = match spec {
        WavSpec {
            sample_format: SampleFormat::Int,
            bits_per_sample: 16,
            channels,
            sample_rate,
        } if sample_rate == target_sample_rate => {
            trace!("16-bit audio at target rate, passing through");
            reader
                .samples::<i16>()
                .step_by(channels as usize)
                .collect::<Result<Vec<i16>, _>>()?
        }
        WavSpec {
            sample_format: SampleFormat::Int,
            bits_per_sample,
            channels,
            sample_rate,
        } => {
            // TODO: handle bits_per_sample=8 which is unsigned

            let magnitude = (bits_per_sample as f32 - 1.0).exp2();
            let samples = reader
                // take all possible bit depths
                .samples::<i32>()
                .step_by(channels as usize)
                .map(|s| s.map(|s| s as f32 / magnitude))
                .collect::<Result<Vec<f32>, _>>()?;

            if sample_rate == target_sample_rate {
                samples.into_iter().map(quantize).collect()
            } else {
                resample_and_quantize(samples, sample_rate, target_sample_rate)?
            }
        }
        WavSpec {
            sample_format: SampleFormat::Float,
            bits_per_sample: 32,
            channels,
            sample_rate,
        } => {
            let samples = reader
                .samples::<f32>()
                .step_by(channels as usize)
                .collect::<Result<Vec<f32>, _>>()?;

            if sample_rate == target_sample_rate {
                samples.into_iter().map(quantize).collect()
            } else {
                resample_and_quantize(samples, sample_rate, target_sample_rate)?
            }
        }
        spec => return Err(AudioError::Unsupported(spec)),
    };

    let mut wave = BytesMut::with_capacity(2 * (samples.len()));
    for s in samples {
        wave.put_i16_le(s);
    }

    Ok(wave.freeze())
}
