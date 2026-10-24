"""Step 1: train on your own hardware and watch the run live on the Ultralytics Platform.

Log in once with `yolo login YOUR_API_KEY` and put your Platform username in PROJECT.
The same run from the CLI: yolo train model=yolo26n.pt data=coco8.yaml epochs=10 project=your_username/gdg-samsun
"""

from ultralytics import YOLO

from device_utils import DEVICE

PROJECT = "your_username/gdg-samsun"  # username/project streams the run to the Ultralytics Platform

model = YOLO("yolo26n.pt")
model.train(data="coco8.yaml", epochs=10, device=DEVICE, project=PROJECT, name="laptop-run")
