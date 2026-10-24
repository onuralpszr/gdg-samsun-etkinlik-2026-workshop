"""Shared device picker so every workshop script runs on NVIDIA, Intel, Apple or a plain CPU."""

DEVICE_HELP = "auto (default), cpu, 0 for CUDA, xpu for Intel GPU, mps for Apple, intel:cpu, intel:gpu or intel:npu"


def pick_device(device, model=""):
    """Turn 'auto' into the best PyTorch accelerator on this machine.

    Exported models such as ONNX or OpenVINO pick their own runtime, so 'auto' gives them 'cpu'.
    Pass intel:gpu or intel:npu explicitly to run OpenVINO on those devices.
    """
    if device != "auto":
        return device
    if model and not str(model).endswith(".pt"):
        return "cpu"
    import torch

    if torch.cuda.is_available():
        return "0"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
