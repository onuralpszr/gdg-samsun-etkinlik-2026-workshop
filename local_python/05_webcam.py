"""Live webcam with an FPS counter, the crowd favorite. Press q to quit.

Pick another camera with CAMERA=1. Swap engines from the CLI:
yolo predict model=yolo26n_openvino_model device=intel:gpu source=0 show=True
"""

import os

import cv2
from ultralytics import YOLO

from device_utils import DEVICE

model = YOLO("yolo26n.pt")
camera = cv2.VideoCapture(int(os.getenv("CAMERA", "0")))
while (frame := camera.read()[1]) is not None:
    result = model(frame, device=DEVICE, verbose=False)[0]
    fps = 1000 / sum(result.speed.values())
    annotated = result.plot()
    cv2.putText(annotated, f"{fps:.0f} FPS on {DEVICE}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, (52, 168, 83), 3)
    cv2.imshow("No GPU Left Behind", annotated)
    if cv2.waitKey(1) == ord("q"):
        break
camera.release()
