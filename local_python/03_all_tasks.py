"""Seven tasks, one line each: the Ultralytics YOLO26 family tour.

detect, segment, semantic, depth, classify, pose and obb all share the exact same API.
Swap the model name and you swap the superpower.

Usage:
    python local_python/03_all_tasks.py                          # all seven tasks
    python local_python/03_all_tasks.py --tasks detect pose      # just a few
    python local_python/03_all_tasks.py --device xpu --save-dir docs/assets/img
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

BUS = "https://ultralytics.com/images/bus.jpg"
ZIDANE = "https://ultralytics.com/images/zidane.jpg"
BOATS = "https://ultralytics.com/images/boats.jpg"

TASKS = {
    "detect": ("yolo26n.pt", BUS),  # boxes around 80 COCO classes
    "segment": ("yolo26n-seg.pt", BUS),  # pixel masks for every object
    "semantic": ("yolo26n-sem.pt", BUS),  # a class label for every pixel
    "depth": ("yolo26n-depth.pt", BUS),  # how far away is every pixel
    "classify": ("yolo26n-cls.pt", BUS),  # one label for the whole image
    "pose": ("yolo26n-pose.pt", ZIDANE),  # 17 body keypoints per person
    "obb": ("yolo26n-obb.pt", BOATS),  # rotated boxes for aerial images
}


def main():
    parser = argparse.ArgumentParser(description="Run every Ultralytics YOLO26 task on a sample image")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS), help="tasks to run")
    parser.add_argument("--device", default="cpu", help="cpu, xpu, mps or 0 for CUDA")
    parser.add_argument("--save-dir", default="runs/tasks", help="folder for the annotated images")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        weights, source = TASKS[task]
        model = YOLO(weights)
        result = model.predict(source, device=args.device, verbose=False)[0]
        out = save_dir / f"task_{task}.jpg"
        busy = task == "obb"  # aerial scenes hold hundreds of boats, so hide labels to keep it readable
        result.save(filename=str(out), labels=not busy, conf=not busy)
        speed = result.speed["inference"]
        print(f"{task:<9} {weights:<18} {speed:6.1f} ms  {result.verbose().strip(', ') or 'done'}  saved {out}")


if __name__ == "__main__":
    main()
