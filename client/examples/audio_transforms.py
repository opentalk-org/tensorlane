import torch


def transform_audio(wave: torch.Tensor) -> torch.Tensor:
    return wave.to(torch.float32) / 32768.0
