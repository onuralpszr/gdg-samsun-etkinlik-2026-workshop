"""Multi object tracking with Ultralytics YOLO and the `botsort` or `bytetrack` trackers.

Every person gets a stable ID that follows them across frames. Works with any detect, segment, pose or obb model.

Usage:
    python local_python/04_track.py                                   # sample crowd video
    python local_python/04_track.py --source 0 --show                 # your webcam
    python local_python/04_track.py --model yolo26n-pose.pt --tracker bytetrack
    python local_python/04_track.py --model yolo26n_openvino_model --device intel:gpu
"""

import argparse
import time

from ultralytics import YOLO

VIDEO = "https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4"


def main():
    parser = argparse.ArgumentParser(description="Track objects in a video, stream or webcam")
    parser.add_argument("--model", default="yolo26n.pt", help="any Ultralytics YOLO model or export")
    parser.add_argument("--source", default=VIDEO, help="video path, URL, RTSP stream or webcam index like 0")
    parser.add_argument("--tracker", default="botsort", choices=["botsort", "bytetrack"], help="tracking algorithm")
    parser.add_argument("--device", default="cpu", help="cpu, xpu, 0, intel:gpu or intel:npu")
    parser.add_argument("--classes", type=int, nargs="*", default=[0], help="class ids to track, 0 is person")
    parser.add_argument("--show", action="store_true", help="open a live window")
    args = parser.parse_args()

    model = YOLO(args.model)
    stream = model.track(
        source=args.source,
        tracker=f"{args.tracker}.yaml",
        device=args.device,
        classes=args.classes,
        stream=True,  # yields one frame at a time so memory stays flat on long videos
        persist=True,
        show=args.show,
        save=True,
        project="gdg-samsun",
        name="track",
        exist_ok=True,
        verbose=False,
    )

    seen, frames, start = set(), 0, time.perf_counter()
    for result in stream:
        frames += 1
        if result.boxes is not None and result.boxes.id is not None:
            seen.update(result.boxes.id.int().tolist())
    fps = frames / (time.perf_counter() - start)
    print(f"Tracked {len(seen)} unique objects across {frames} frames at {fps:.1f} FPS with {args.tracker}")
    print(f"Annotated video saved in {model.predictor.save_dir}")


if __name__ == "__main__":
    main()
