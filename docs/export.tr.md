---
title: "Adım 2: Dışa aktarma ve büyük hız yarışı"
---

# Adım 2: Dışa aktarma ve büyük hız yarışı :package:

```bash
# her yerde çalışır
yolo export model=yolo26n.pt format=onnx simplify=True

# Intel CPU, GPU ve NPU için
yolo export model=yolo26n.pt format=openvino

# NVIDIA TensorRT FP16
yolo export model=yolo26n.pt format=engine half=True

# dışa aktarır, sonra her formatı her cihazda yarıştırır
python local_python/02_export_model.py
```

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="onnx", simplify=True)  # her yerde çalışır
model.export(format="openvino")  # Intel CPU, GPU ve NPU için
model.export(format="engine", half=True)  # NVIDIA TensorRT FP16

YOLO("yolo26n.pt").predict("bus.jpg", device="cpu")  # her CPU üzerinde PyTorch
YOLO("yolo26n.pt").predict("bus.jpg", device="mps")  # Apple silicon üzerinde PyTorch
YOLO("yolo26n.onnx").predict("bus.jpg", device=0)  # NVIDIA GPU üzerinde ONNX Runtime
YOLO("yolo26n.engine").predict("bus.jpg", device=0)  # NVIDIA GPU üzerinde TensorRT
YOLO("yolo26n_openvino_model").predict("bus.jpg", device="intel:gpu")  # Intel GPU üzerinde OpenVINO
YOLO("yolo26n_openvino_model").predict("bus.jpg", device="intel:npu")  # Intel NPU üzerinde OpenVINO
```

!!! info "NVIDIA GPU'nuz varsa"

    Yarış CUDA bulduğunda kendiliğinden bir TensorRT FP16 satırı da ekler. Colab'da T4 çalışma ortamını seçerseniz notebook bu adımı sizin için yapar. Rust da ONNX dosyasından doğrudan TensorRT FP16 çalıştırabilir, ayrıntılar [Adım 3.4](rust.md#34-onnx-uzerinden-dogrudan-nvidia-tensorrt-fp16) bölümünde.

[:material-arrow-right: Sonraki: Adım 3 Rust](rust.md){ .md-button .md-button--primary .next-step }
