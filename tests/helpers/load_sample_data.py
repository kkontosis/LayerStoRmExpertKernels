"""Utilities to load pre-generated sample tensors for testing."""

import os
import torch

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "sample-data")


def load_sample(name: str) -> torch.Tensor:
    """Load a .pt tensor from sample-data/."""
    path = os.path.join(SAMPLE_DATA_DIR, f"{name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Sample data not found: {path}\n"
            f"Run: python sample-data/generate_samples.py"
        )
    return torch.load(path, weights_only=True)
