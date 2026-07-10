"""Shared helpers."""

import torch


def get_device() -> torch.device:
    """
    Best available torch device, in order of preference:
        1. CUDA  - NVIDIA GPU (Linux/Windows machines with CUDA drivers)
        2. MPS   - Apple Silicon GPU (Metal, macOS)
        3. CPU   - universal fallback if neither GPU backend is available

    Every model-loading module in this project should call this instead
    of hardcoding a cuda-or-cpu check, so the same code runs at full
    speed on an NVIDIA workstation, an Apple Silicon laptop, or falls
    back safely to CPU-only hardware.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
