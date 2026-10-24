"""Seven tasks, one API: swap the model name and you swap the task.

The same from the CLI: yolo predict model=yolo26n-pose.pt source=https://ultralytics.com/images/zidane.jpg
"""

from ultralytics import YOLO

from device_utils import DEVICE

BUS = "https://ultralytics.com/images/bus.jpg"
TASKS = {
    "detect": ("yolo26n.pt", BUS),  # boxes around 80 COCO classes
    "segment": ("yolo26n-seg.pt", BUS),  # a mask for every object
    "semantic": ("yolo26n-sem.pt", BUS),  # a class for every pixel
    "depth": ("yolo26n-depth.pt", BUS),  # how far away every pixel is
    "classify": ("yolo26n-cls.pt", BUS),  # one label for the whole image
    "pose": ("yolo26n-pose.pt", "https://ultralytics.com/images/zidane.jpg"),  # 17 keypoints per person
    "obb": ("yolo26n-obb.pt", "https://ultralytics.com/images/boats.jpg"),  # rotated boxes for aerial images
}

for task, (weights, image) in TASKS.items():
    result = YOLO(weights)(image, device=DEVICE, verbose=False)[0]
    result.save(f"runs/tasks/task_{task}.jpg", labels=task != "obb")  # hundreds of ships, so obb skips the labels
    print(f"{task:<9} {result.speed['inference']:5.1f} ms  {result.verbose().rstrip(', ')}")
