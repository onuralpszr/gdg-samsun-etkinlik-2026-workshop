"""Live webcam demo with an FPS counter. The crowd favorite.

Point the camera at the audience, then switch between PyTorch and OpenVINO to feel the speed difference.
Press q to quit.

Usage:
    python local_python/05_webcam.py                                            # best device on this machine
    python local_python/05_webcam.py --model yolo26n_openvino_model --device intel:gpu
    python local_python/05_webcam.py --model yolo26n-pose.pt --track
    python local_python/05_webcam.py --model yolo26n-seg.pt --camera 1
"""

import argparse
import time

import cv2
from ultralytics import YOLO

from device_utils import DEVICE_HELP, pick_device


def main():
    parser = argparse.ArgumentParser(description="Live Ultralytics YOLO on your webcam")
    parser.add_argument("--model", default="yolo26n.pt", help="any Ultralytics YOLO model or export")
    parser.add_argument("--device", default="auto", help=DEVICE_HELP)
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--track", action="store_true", help="add tracking IDs")
    parser.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after this many frames, 0 runs forever")
    parser.add_argument("--no-show", action="store_true", help="run without a window, handy over SSH")
    args = parser.parse_args()
    args.device = pick_device(args.device, args.model)

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try --camera 1")

    frames, fps, tick = 0, 0.0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.track:
            result = model.track(frame, persist=True, device=args.device, conf=args.conf, verbose=False)[0]
        else:
            result = model.predict(frame, device=args.device, conf=args.conf, verbose=False)[0]

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - tick, 1e-6)  # smoothed so the number does not jump around
        tick = now
        frames += 1

        annotated = result.plot()
        label = f"{fps:.0f} FPS  {args.device}  {len(result)} objects"
        cv2.putText(annotated, label, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (52, 168, 83), 3)
        if not args.no_show:
            cv2.imshow("No GPU Left Behind", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if args.max_frames and frames >= args.max_frames:
            break

    cap.release()
    if not args.no_show:
        cv2.destroyAllWindows()
    print(f"Processed {frames} frames, last smoothed speed {fps:.1f} FPS on {args.device}")


if __name__ == "__main__":
    main()
