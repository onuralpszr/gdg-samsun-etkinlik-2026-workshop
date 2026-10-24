"""Real world use cases in a few lines with Ultralytics Solutions.

Pick one: python local_python/06_solutions.py heatmap   (count, heatmap, blur or speed)
Or straight from the CLI: yolo solutions heatmap source=video.mp4 show=True
"""

import sys
from pathlib import Path

import cv2
from ultralytics import solutions
from ultralytics.utils.downloads import safe_download

from device_utils import DEVICE

VIDEO = "https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4"

name = sys.argv[1] if len(sys.argv) > 1 else "count"
video = safe_download(VIDEO, dir="videos")
cap = cv2.VideoCapture(str(video))
w, h, fps = (int(cap.get(p)) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
line = [(0, h // 2), (w, h // 2)]  # counting and speed line across the middle

solution_class, extra = {
    "count": (solutions.ObjectCounter, {"region": line}),
    "heatmap": (solutions.Heatmap, {"colormap": cv2.COLORMAP_PARULA}),
    "blur": (solutions.ObjectBlurrer, {"blur_ratio": 0.5}),
    "speed": (solutions.SpeedEstimator, {"region": line}),
}[name]
solution = solution_class(model="yolo26n.pt", device=DEVICE, classes=[0], show=False, verbose=False, **extra)

output = Path(f"runs/solutions/{name}.mp4")
output.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
while (frame := cap.read()[1]) is not None:
    writer.write(solution(frame).plot_im)
writer.release()
print(f"Saved {output}")
