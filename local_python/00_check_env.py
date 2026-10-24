"""Prerequisites check for the No GPU Left Behind workshop.

Prints every accelerator this machine can use so you know which device to pass to Ultralytics YOLO.

Usage:
    python local_python/00_check_env.py
"""

import os
import platform
import shutil
import sys


def check(label, ok, detail=""):
    """Print one tidy status line."""
    mark = "OK  " if ok else "MISS"
    print(f"[{mark}] {label:<28} {detail}")


def main():
    print(f"\nNo GPU Left Behind prerequisites check on {platform.node()} ({platform.system()} {platform.machine()})\n")
    check("Python", sys.version_info >= (3, 11), platform.python_version())

    import torch

    check("PyTorch", True, torch.__version__)
    check("NVIDIA CUDA", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    check("Intel XPU (PyTorch)", has_xpu, torch.xpu.get_device_name(0) if has_xpu else "")
    has_mps = torch.backends.mps.is_available()
    check("Apple MPS", has_mps, "Apple Silicon GPU" if has_mps else "")

    import ultralytics

    check("Ultralytics YOLO", True, ultralytics.__version__)

    try:
        import openvino as ov

        devices = ov.Core().available_devices
        check("OpenVINO", True, f"{ov.__version__.split('-')[0]} devices: {', '.join(devices)}")
        check("Intel NPU (OpenVINO)", "NPU" in devices, "ready for intel:npu" if "NPU" in devices else "")
    except ImportError:
        check("OpenVINO", False, "pip install openvino")

    try:
        import onnxruntime as ort

        check("ONNX Runtime", True, f"{ort.__version__} providers: {', '.join(ort.get_available_providers())}")
    except ImportError:
        check("ONNX Runtime", False, "pip install onnxruntime")

    check("Rust cargo", shutil.which("cargo") is not None, "needed for Step 3")
    check("ultralytics-inference CLI", shutil.which("ultralytics-inference") is not None, "cargo install ultralytics-inference")

    from ultralytics.utils import SETTINGS

    has_key = bool(os.getenv("ULTRALYTICS_API_KEY") or SETTINGS.get("api_key"))
    check("Ultralytics Platform key", has_key, "logged in" if has_key else "run: yolo login YOUR_API_KEY")

    best = "0" if torch.cuda.is_available() else "xpu" if has_xpu else "mps" if has_mps else "cpu"
    print(f"\nRecommended training device: --device {best}")
    print("Recommended fast inference: OpenVINO on intel:gpu or intel:npu when you have Intel silicon\n")


if __name__ == "__main__":
    main()
