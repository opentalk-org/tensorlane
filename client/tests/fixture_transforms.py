import time
import os
import torch


def double(wave):
    if wave.dtype != torch.int16 or wave.device.type != "cpu" or wave.ndim != 1:
        raise TypeError("expected raw one-dimensional CPU int16 input")
    return wave.to(torch.int64) * 2


def fail(wave):
    raise ValueError("fixture failure")


def slow(wave):
    time.sleep(60)
    return wave


def invalid(wave):
    return None


def identify_worker(wave):
    if int(wave[0]) == 0:
        time.sleep(0.3)
    return torch.tensor([int(wave[0]), os.getpid()])
