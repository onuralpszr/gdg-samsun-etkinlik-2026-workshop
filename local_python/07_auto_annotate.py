"""Automatic annotation: let Ultralytics YOLO find objects and SAM draw the masks.

Labeling by hand is slow. Point this at a folder of images and get YOLO segmentation labels for free,
then review them in Ultralytics Platform (which also has Smart Annotation with SAM built in).

Usage:
    python local_python/07_auto_annotate.py                                 # sample images
    python local_python/07_auto_annotate.py --images path/to/your/images --classes 0 2
    python local_python/07_auto_annotate.py --sam sam2.1_t.pt --device xpu
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics.data.annotator import auto_annotate
from ultralytics.utils.downloads import safe_download
from ultralytics.utils.plotting import colors

SAMPLES = ["https://ultralytics.com/images/bus.jpg", "https://ultralytics.com/images/zidane.jpg"]


def preview(image_path, label_path, names, out_path):
    """Draw the generated polygons on the image so you can eyeball the labels."""
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    overlay = image.copy()
    for line in label_path.read_text().splitlines():
        values = line.split()
        cls, points = int(values[0]), np.array(values[1:], dtype=float).reshape(-1, 2)
        polygon = (points * [w, h]).astype(np.int32)
        color = colors(cls, True)
        cv2.fillPoly(overlay, [polygon], color)
        cv2.polylines(image, [polygon], True, color, 2)
        x, y = (int(v) for v in polygon[polygon[:, 1].argmin()])  # anchor the tag at the top of the mask
        text = names.get(cls, str(cls))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        y = max(y, th + 10)
        cv2.rectangle(image, (x, y - th - 10), (x + tw + 8, y), color, -1)
        cv2.putText(image, text, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(out_path), cv2.addWeighted(overlay, 0.4, image, 0.6, 0))


def main():
    parser = argparse.ArgumentParser(description="Auto annotate images with Ultralytics YOLO plus SAM")
    parser.add_argument("--images", default="datasets/auto_annotate_demo/images", help="folder with images")
    parser.add_argument("--det", default="yolo26n.pt", help="Ultralytics YOLO detection model")
    parser.add_argument("--sam", default="sam2.1_b.pt", help="SAM model: sam2.1_t.pt, sam2.1_b.pt, mobile_sam.pt")
    parser.add_argument("--device", default="cpu", help="cpu, xpu, mps or 0 for CUDA")
    parser.add_argument("--classes", type=int, nargs="*", default=None, help="only label these class ids")
    parser.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    args = parser.parse_args()

    images = Path(args.images)
    if not images.exists():
        images.mkdir(parents=True)
        for url in SAMPLES:
            safe_download(url, dir=images)

    labels = images.parent / "labels"
    auto_annotate(
        data=images,
        det_model=args.det,
        sam_model=args.sam,
        device=args.device,
        conf=args.conf,
        classes=args.classes,
        output_dir=labels,
    )

    from ultralytics import YOLO

    names = YOLO(args.det).names
    previews = Path("runs/auto_annotate")
    previews.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(images.glob("*.jpg")):
        label_path = labels / f"{image_path.stem}.txt"
        if label_path.exists():
            out = previews / f"auto_{image_path.stem}.jpg"
            preview(image_path, label_path, names, out)
            count = len(label_path.read_text().splitlines())
            print(f"{image_path.name}: {count} polygons written to {label_path}, preview {out}")
    print("Upload the images and labels folders to Ultralytics Platform to review and train")


if __name__ == "__main__":
    main()
