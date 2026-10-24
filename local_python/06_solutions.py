"""Real world use cases in a few lines with Ultralytics Solutions.

count     people crossing a line (retail, events, turnstiles)
heatmap   where people spend their time (store layouts, museums)
blur      privacy blur on every person (GDPR friendly recordings)
speed     estimated speed of every moving object (traffic, sports)

Usage:
    python local_python/06_solutions.py --solution count
    python local_python/06_solutions.py --solution heatmap --device intel:gpu --model yolo26n_openvino_model
    python local_python/06_solutions.py --solution blur --source 0
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import solutions
from ultralytics.utils.downloads import safe_download

from device_utils import DEVICE_HELP, pick_device

VIDEO = "https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4"


def build(name, model, device, width, height):
    """Create the requested solution with a counting line across the middle of the frame."""
    line = [(0, height // 2), (width, height // 2)]
    common = {"model": model, "device": device, "show": False, "verbose": False, "classes": [0]}
    if name == "count":
        return solutions.ObjectCounter(region=line, **common)
    if name == "heatmap":
        return solutions.Heatmap(colormap=cv2.COLORMAP_PARULA, **common)
    if name == "blur":
        return solutions.ObjectBlurrer(blur_ratio=0.5, **common)
    return solutions.SpeedEstimator(region=line, **common)


def main():
    parser = argparse.ArgumentParser(description="Ultralytics Solutions demos")
    parser.add_argument("--solution", default="count", choices=["count", "heatmap", "blur", "speed"])
    parser.add_argument("--model", default="yolo26n.pt", help="any Ultralytics YOLO model or export")
    parser.add_argument("--source", default=VIDEO, help="video path, URL or webcam index like 0")
    parser.add_argument("--device", default="auto", help=DEVICE_HELP)
    parser.add_argument("--output", default=None, help="output video path, runs/solutions/<name>.mp4 by default")
    args = parser.parse_args()
    args.device = pick_device(args.device, args.model)

    source = args.source
    if source.startswith("http"):
        source = Path("videos") / Path(source).name
        if not source.exists():
            safe_download(args.source, file=source)
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {source}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    output = Path(args.output or f"runs/solutions/{args.solution}.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    solution = build(args.solution, args.model, args.device, w, h)

    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = solution(frame)
        writer.write(result.plot_im)
        frames += 1

    cap.release()
    writer.release()
    summary = {
        "count": f"in {result.in_count} out {result.out_count}",
        "speed": f"{result.total_tracks} objects in the last frame",
    }
    print(f"{args.solution}: {frames} frames, {summary.get(args.solution, 'done')}, saved {output}")


if __name__ == "__main__":
    main()
