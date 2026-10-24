"""Step 0: what can this machine run? For the full system report run `yolo checks`."""

import os
import shutil

import openvino as ov
import torch
import ultralytics
from ultralytics.utils import SETTINGS

from device_utils import DEVICE

cuda, xpu, mps = torch.cuda.is_available(), torch.xpu.is_available(), torch.backends.mps.is_available()
print(f"Ultralytics YOLO {ultralytics.__version__} with PyTorch {torch.__version__}")
print(f"NVIDIA CUDA {cuda}, Intel XPU {xpu}, Apple MPS {mps}")
print(f"OpenVINO devices: {', '.join(ov.Core().available_devices)}")
print(f"Rust CLI: {'ready' if shutil.which('ultralytics-inference') else 'cargo install ultralytics-inference'}")
print(f"Platform key: {'saved' if SETTINGS.get('api_key') or os.getenv('ULTRALYTICS_API_KEY') else 'run yolo login'}")
print(f"Best PyTorch device here: {DEVICE}")
