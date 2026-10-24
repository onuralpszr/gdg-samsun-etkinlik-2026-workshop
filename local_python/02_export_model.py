"""Step 2: export Ultralytics YOLO to ONNX and OpenVINO, then race every format on every device.

The export alone is one CLI line: yolo export model=yolo26n.pt format=openvino
"""

from pathlib import Path
from statistics import median

import openvino as ov
from ultralytics import YOLO
from ultralytics.utils import ASSETS

from device_utils import DEVICE

WEIGHTS = "yolo26n.pt"  # or your own runs/detect/.../weights/best.pt

model = YOLO(WEIGHTS)
onnx = model.export(format="onnx", simplify=True)
openvino = model.export(format="openvino")

racers = [(WEIGHTS, "cpu"), (WEIGHTS, DEVICE), (onnx, "cpu")]
racers += [(openvino, f"intel:{name.split('.')[0].lower()}") for name in ov.Core().available_devices]
if DEVICE == "0":
    racers.append((model.export(format="engine", half=True), "0"))  # TensorRT FP16 on NVIDIA

for weights, device in dict.fromkeys(racers):  # dict.fromkeys drops duplicates and keeps the order
    racer = YOLO(weights, task="detect")
    speeds = [racer(ASSETS / "bus.jpg", device=device, verbose=False)[0].speed for _ in range(110)][10:]
    inference = median(s["inference"] for s in speeds)
    total = median(sum(s.values()) for s in speeds)  # preprocess plus inference plus postprocess
    name = Path(weights).name
    print(f"{name:<24} {device:<10} {inference:6.2f} ms inference {total:6.2f} ms end to end {1000 / total:4.0f} FPS")
