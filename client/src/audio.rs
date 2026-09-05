use anyhow::{Context, ensure};
use tch::{Device, Kind, Tensor};

const N_FFT: i64 = 2048;
const WIN_LENGTH: usize = 1200;
const HOP_LENGTH: i64 = 300;
const N_MELS: usize = 80;
const SAMPLE_RATE: f64 = 16_000.0;
const EDGE_PAD_SAMPLES: usize = 5_000;
const MIN_WAVE_SAMPLES: usize = 24_600;

pub struct ProcessedSample {
    pub wave: Tensor,
    pub mel: Tensor,
    pub duration: f64,
    pub speaker_id: i64,
    pub language_id: i64,
    pub text: Tensor,
}

pub struct AudioProcessor {
    window: Tensor,
    filterbank: Tensor,
}

impl AudioProcessor {
    pub fn new() -> anyhow::Result<Self> {
        let window = (0..WIN_LENGTH)
            .map(|index| {
                let phase = std::f32::consts::TAU * index as f32 / WIN_LENGTH as f32;
                0.5 - 0.5 * phase.cos()
            })
            .collect::<Vec<_>>();
        Ok(Self {
            window: Tensor::f_from_slice(&window)?,
            filterbank: Tensor::f_from_slice(&mel_filterbank()?)?
                .f_reshape([N_MELS as i64, N_FFT / 2 + 1])?,
        })
    }

    pub fn process(
        &self,
        wave_bytes: &[u8],
        text_bytes: &[u8],
        duration: f64,
        speaker_id: i64,
        language_id: i32,
    ) -> anyhow::Result<ProcessedSample> {
        let wave = decode_wave(wave_bytes)?;
        let text = decode_text(text_bytes)?;
        let mel = self.mel(&wave)?;
        Ok(ProcessedSample {
            wave,
            mel,
            duration,
            speaker_id,
            language_id: i64::from(language_id),
            text,
        })
    }

    fn mel(&self, wave: &Tensor) -> anyhow::Result<Tensor> {
        let centered = wave
            .f_unsqueeze(0)?
            .f_reflection_pad1d([N_FFT / 2, N_FFT / 2])?
            .f_squeeze_dim(0)?;
        let spectrum = centered.f_stft(
            N_FFT,
            Some(HOP_LENGTH),
            Some(WIN_LENGTH as i64),
            Some(&self.window),
            false,
            true,
            true,
            false,
        )?;
        let power = spectrum.f_abs()?.f_square()?;
        let mel = self.filterbank.f_matmul(&power)?;
        let normalized = mel
            .f_add_scalar(1e-5)?
            .f_log()?
            .f_add_scalar(4.0)?
            .f_div_scalar(4.0)?;
        let frames = normalized
            .size()
            .get(1)
            .copied()
            .context("mel tensor has no time dimension")?;
        Ok(normalized.f_narrow(1, 0, frames - frames % 2)?)
    }
}

fn decode_wave(bytes: &[u8]) -> anyhow::Result<Tensor> {
    ensure!(
        bytes.len().is_multiple_of(2),
        "wave byte count is not divisible by two"
    );
    let values = bytes
        .chunks_exact(2)
        .map(|chunk| {
            let encoded: [u8; 2] = chunk.try_into().context("decoding int16 wave sample")?;
            anyhow::Ok(i16::from_le_bytes(encoded))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let wave = Tensor::f_from_slice(&values)?
        .f_to_kind(Kind::Float)?
        .f_div_scalar(32768.0)?;
    let padded_length = values.len() + EDGE_PAD_SAMPLES * 2;
    let missing = MIN_WAVE_SAMPLES.saturating_sub(padded_length);
    let left = EDGE_PAD_SAMPLES + missing / 2;
    let right = EDGE_PAD_SAMPLES + missing - missing / 2;
    let left = Tensor::f_zeros([left as i64], (Kind::Float, Device::Cpu))?;
    let right = Tensor::f_zeros([right as i64], (Kind::Float, Device::Cpu))?;
    Tensor::f_cat(&[left, wave, right], 0).context("padding waveform")
}

fn decode_text(bytes: &[u8]) -> anyhow::Result<Tensor> {
    ensure!(
        bytes.len().is_multiple_of(8),
        "text byte count is not divisible by eight"
    );
    let values = bytes
        .chunks_exact(8)
        .map(|chunk| {
            let encoded: [u8; 8] = chunk.try_into().context("decoding text token")?;
            anyhow::Ok(i64::from_le_bytes(encoded))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Tensor::f_from_slice(&values).context("creating text tensor")
}

fn mel_filterbank() -> anyhow::Result<Vec<f32>> {
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(SAMPLE_RATE / 2.0);
    let mel_points = (0..N_MELS + 2)
        .map(|index| {
            let ratio = index as f64 / (N_MELS + 1) as f64;
            mel_to_hz(mel_min + ratio * (mel_max - mel_min))
        })
        .collect::<Vec<_>>();
    let frequencies = (0..=N_FFT / 2)
        .map(|index| index as f64 * SAMPLE_RATE / N_FFT as f64)
        .collect::<Vec<_>>();
    let mut output = Vec::with_capacity(N_MELS * frequencies.len());
    for points in mel_points.windows(3) {
        let &[left, center, right] = points
            .try_into()
            .context("mel filter window does not have three points")?;
        output.extend(frequencies.iter().map(|frequency| {
            let lower = (frequency - left) / (center - left);
            let upper = (right - frequency) / (right - center);
            lower.min(upper).max(0.0) as f32
        }));
    }
    Ok(output)
}

fn hz_to_mel(frequency: f64) -> f64 {
    2595.0 * (1.0 + frequency / 700.0).log10()
}

fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10_f64.powf(mel / 2595.0) - 1.0)
}
