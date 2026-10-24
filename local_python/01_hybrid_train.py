"""Step 1: Hybrid training with Ultralytics YOLO and the Ultralytics Platform.

Your laptop does the training. The Ultralytics Platform gets live metrics, plots, console logs and the final
checkpoint so the whole room can watch your run in the browser. No API key? Training still works fully offline.

Usage:
    export ULTRALYTICS_API_KEY=ul_xxx      # or run once: yolo login ul_xxx
    python local_python/01_hybrid_train.py --model yolo26n.pt --data coco8.yaml --epochs 3 --device xpu --project gdg-samsun
    python local_python/01_hybrid_train.py --data ul://your_username/datasets/your_dataset --project gdg-samsun
    python local_python/01_hybrid_train.py --local-only --device cpu
"""

import argparse
import os
import time

from ultralytics import YOLO
from ultralytics.utils import PLATFORM_URL, SETTINGS


def parse_args():
    """Read command line flags."""
    parser = argparse.ArgumentParser(description="Train Ultralytics YOLO locally and stream it to Ultralytics Platform")
    parser.add_argument("--model", default="yolo26n.pt", help="model weights or yaml, yolo26n.pt by default")
    parser.add_argument("--data", default="coco8.yaml", help="dataset yaml or ul://user/datasets/name Platform URI")
    parser.add_argument("--epochs", type=int, default=3, help="number of training epochs")
    parser.add_argument("--device", default="auto", help="auto, cpu, xpu, mps, 0 or 0,1 for CUDA")
    parser.add_argument("--project", default="gdg-samsun", help="Platform project name or user/project")
    parser.add_argument("--name", default="laptop-run", help="run name shown on the Platform")
    parser.add_argument("--imgsz", type=int, default=640, help="training image size")
    parser.add_argument("--batch", type=int, default=16, help="batch size, use -1 for auto batch on CUDA")
    parser.add_argument("--api-key", default=None, help="Ultralytics Platform API key, defaults to ULTRALYTICS_API_KEY")
    parser.add_argument("--local-only", action="store_true", help="skip the Platform and keep everything on disk")
    return parser.parse_args()


def pick_device(device):
    """Turn 'auto' into the best accelerator on this machine."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "0"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def platform_login(api_key):
    """Validate the API key with the Ultralytics Platform and return the username, or None when offline."""
    api_key = api_key or os.getenv("ULTRALYTICS_API_KEY") or SETTINGS.get("api_key")
    if not api_key:
        print(f"No Ultralytics Platform key found. Grab one at {PLATFORM_URL}/settings?tab=api-keys")
        print("Training continues locally. Your results will be saved under runs/")
        return None

    from ultralytics import Platform

    try:
        with Platform(api_key=api_key, base_url=PLATFORM_URL, timeout=30) as client:
            account = client.account.summary()
    except Exception as error:
        print(f"Ultralytics Platform login failed ({error}). Training continues locally.")
        return None

    os.environ["ULTRALYTICS_API_KEY"] = api_key  # the Platform callback reads this during training
    credits = account.get("creditsCents", 0) / 100
    print(
        f"Logged in to Ultralytics Platform as {account['username']} ({account['plan']} plan, ${credits:.2f} credits)"
    )
    return account["username"]


def main():
    args = parse_args()
    device = pick_device(args.device)
    username = None if args.local_only else platform_login(args.api_key)

    project = args.project
    if username and "/" not in project:
        project = f"{username}/{project}"  # user/project tells Ultralytics YOLO to stream to the Platform
    if args.local_only:
        project = None  # the Platform callback only streams when a project is set

    print(f"Training {args.model} on {args.data} for {args.epochs} epochs on device '{device}'")
    model = YOLO(args.model)
    start = time.perf_counter()
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=project,
        name=args.name,
        exist_ok=True,
        plots=True,
    )
    minutes = (time.perf_counter() - start) / 60

    save_dir = model.trainer.save_dir
    metrics = model.trainer.metrics
    print(f"\nDone in {minutes:.1f} minutes on '{device}'")
    print(f"Best weights: {save_dir / 'weights' / 'best.pt'}")
    print(f"mAP50 {metrics.get('metrics/mAP50(B)', 0):.3f}  mAP50:95 {metrics.get('metrics/mAP50-95(B)', 0):.3f}")
    if username:
        print(f"Watch it live: {PLATFORM_URL}/{project}")
    print(f"Next step: python local_python/02_export_model.py --weights {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
