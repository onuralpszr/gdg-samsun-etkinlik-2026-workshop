"""The fastest PyTorch device on this machine: NVIDIA CUDA, Intel XPU, Apple MPS or CPU."""

import torch

if torch.cuda.is_available():
    DEVICE = "0"
elif torch.xpu.is_available():
    DEVICE = "xpu"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
