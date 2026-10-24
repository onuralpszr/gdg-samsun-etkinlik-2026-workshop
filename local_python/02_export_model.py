"""Step 2: Export Ultralytics YOLO to ONNX, OpenVINO and TensorRT, then race them.

Every format runs the same image many times and you get a latency, FPS and file size table at the end.
OpenVINO runs on every Intel device it can find (CPU, GPU and NPU). TensorRT joins the race when CUDA is present.

Usage:
    python local_python/02_export_model.py                                  # yolo26n.pt, ONNX plus OpenVINO
    python local_python/02_export_model.py --weights runs/detect/gdg-samsun/laptop-run/weights/best.pt
    python local_python/02_export_model.py --tensorrt                       # adds a TensorRT FP16 engine on NVIDIA
    python local_python/02_export_model.py --save-md docs/assets/results/benchmark.md
"""

import argparse
import statistics
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.utils import ASSETS


def parse_args():
    """Read command line flags."""
    parser = argparse.ArgumentParser(description="Export Ultralytics YOLO and compare speed and size")
    parser.add_argument("--weights", default="yolo26n.pt", help="trained .pt checkpoint, yolo26n.pt by default")
    parser.add_argument("--imgsz", type=int, default=640, help="export and inference image size")
    parser.add_argument("--source", default=str(ASSETS / "bus.jpg"), help="image used for the speed race")
    parser.add_argument("--runs", type=int, default=50, help="timed runs per format after warmup")
    parser.add_argument("--tensorrt", action="store_true", help="also export a TensorRT engine with half=True")
    parser.add_argument("--half", action="store_true", help="export OpenVINO with FP16 weights")
    parser.add_argument("--skip-openvino", action="store_true", help="skip the OpenVINO export")
    parser.add_argument("--save-md", default=None, help="write the results table to this markdown file")
    return parser.parse_args()


def size_mb(path):
    """File size in MB, or the total size of a folder such as an OpenVINO export."""
    path = Path(path)
    files = path.rglob("*") if path.is_dir() else [path]
    return sum(f.stat().st_size for f in files if f.is_file()) / 1e6


def race(weights, device, source, imgsz, runs):
    """Return the median inference time in ms and the detection count for one format on one device."""
    model = YOLO(weights, task="detect")
    for _ in range(5):  # warmup lets each backend compile kernels and allocate memory
        model.predict(source, imgsz=imgsz, device=device, verbose=False)
    times = []
    for _ in range(runs):
        result = model.predict(source, imgsz=imgsz, device=device, verbose=False)[0]
        times.append(result.speed["inference"])
    return statistics.median(times), len(result.boxes) if result.boxes is not None else 0


def openvino_devices():
    """Intel devices that OpenVINO can see on this machine."""
    try:
        import openvino as ov
    except ImportError:
        return []
    available = ov.Core().available_devices
    return [f"intel:{name.lower()}" for name in ("CPU", "GPU", "NPU") if any(d.startswith(name) for d in available)]


def main():
    args = parse_args()
    model = YOLO(args.weights)
    candidates = [("PyTorch", args.weights, "cpu")]
    if torch.cuda.is_available():
        candidates.append(("PyTorch", args.weights, "0"))
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        candidates.append(("PyTorch", args.weights, "xpu"))

    print("Exporting ONNX with simplify=True")
    onnx_path = model.export(format="onnx", imgsz=args.imgsz, simplify=True)
    candidates.append(("ONNX", onnx_path, "cpu"))

    if not args.skip_openvino:
        print("Exporting OpenVINO")
        ov_path = model.export(format="openvino", imgsz=args.imgsz, half=args.half)
        candidates += [("OpenVINO", ov_path, device) for device in openvino_devices()]

    if args.tensorrt:
        if torch.cuda.is_available():
            print("Exporting TensorRT with half=True")
            engine_path = model.export(format="engine", imgsz=args.imgsz, half=True, device=0)
            candidates.append(("TensorRT FP16", engine_path, "0"))
        else:
            print("TensorRT needs an NVIDIA GPU with CUDA, skipping it on this machine")

    rows = []
    for fmt, path, device in candidates:
        print(f"Racing {fmt} on {device}")
        try:
            ms, count = race(path, device, args.source, args.imgsz, args.runs)
        except Exception as error:
            print(f"  {fmt} on {device} could not run: {error}")
            continue
        rows.append({"format": fmt, "device": device, "size": size_mb(path), "ms": ms, "objects": count})

    baseline = rows[0]["ms"]
    header = "| Format | Device | Size (MB) | Inference (ms) | FPS | Speedup | Objects |"
    lines = [header, "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['format']} | {r['device']} | {r['size']:.1f} | {r['ms']:.1f} | {1000 / r['ms']:.0f} "
            f"| {baseline / r['ms']:.1f}x | {r['objects']} |"
        )
    table = "\n".join(lines)
    print(f"\n{Path(args.weights).name} at imgsz {args.imgsz}, median of {args.runs} runs\n")
    print(table)
    fastest = min(rows, key=lambda r: r["ms"])
    print(f"\nFastest: {fastest['format']} on {fastest['device']} at {fastest['ms']:.1f} ms")

    if args.save_md:
        out = Path(args.save_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table + "\n")
        print(f"Saved table to {out}")


if __name__ == "__main__":
    main()
