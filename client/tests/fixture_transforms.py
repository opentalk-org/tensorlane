import os
import time
import torch


def transform(sample):
    value = sample.metadata["position"]
    return {
        "value": torch.tensor([value, -value]),
        "nested": [sample.metadata["label"], (sample.blobs["payload"],)],
    }


def collate(samples):
    return {
        "values": torch.stack([sample["value"] for sample in samples]),
        "labels": [sample["nested"][0] for sample in samples],
    }


def fail(sample):
    raise ValueError("fixture transform failure")


def fail_collate(samples):
    raise ValueError("fixture collation failure")


def slow(sample):
    time.sleep(60)
    return sample


def identify_worker(sample):
    if sample.metadata["position"] == 0:
        time.sleep(0.3)
    return {"value": torch.tensor([sample.metadata["position"], os.getpid()])}


def invalid(sample):
    return object()


def timed_transform(sample):
    time.sleep(0.01)
    return transform(sample)
