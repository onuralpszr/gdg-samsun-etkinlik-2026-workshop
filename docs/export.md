---
title: "Step 2: Export and the great speed race"
---

# Step 2: Export and the great speed race :package:

```bash
# runs everywhere
yolo export model=yolo26n.pt format=onnx simplify=True

# Intel CPU, GPU and NPU
yolo export model=yolo26n.pt format=openvino

# NVIDIA TensorRT FP16
yolo export model=yolo26n.pt format=engine half=True

# export, then race every format on every device
python local_python/02_export_model.py
```

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="onnx", simplify=True)  # runs everywhere
model.export(format="openvino")  # Intel CPU, GPU and NPU
model.export(format="engine", half=True)  # NVIDIA TensorRT FP16

YOLO("yolo26n_openvino_model").predict("bus.jpg", device="intel:npu")  # yes, the NPU
```

!!! info "Got an NVIDIA GPU?"

    The race adds a TensorRT FP16 row on its own when it finds CUDA. On Colab pick the T4 runtime and the notebook does it for you. Rust can run TensorRT FP16 straight from the ONNX file too, see [Step 3.4](rust.md#34-nvidia-tensorrt-fp16-straight-from-onnx).

[:material-arrow-right: Next: Step 3 Rust](rust.md){ .md-button .md-button--primary .next-step }
