"""Build the pictures used on the workshop page and README.

Run it from the repo root after the workshop scripts have produced their outputs:
    python local_python/03_all_tasks.py && python local_python/07_auto_annotate.py
    python tools/make_visuals.py
"""

import subprocess
from itertools import product
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from ultralytics.utils.checks import check_font
from ultralytics.utils.plotting import colors

IMG = Path("docs/assets/img")
GDG = ["#4285f4", "#ea4335", "#fbbc04", "#34a853"]

# Frames per second on the workshop laptop, end to end, from the docs tables (None means not measured)
FPS_ROWS = [
    ("PyTorch  cpu", 65, None),
    ("PyTorch  xpu", 174, None),
    ("ONNX Runtime  cpu", 15, 61),
    ("OpenVINO  intel:cpu", 64, 40),
    ("OpenVINO  intel:npu", 220, 366),
    ("OpenVINO  intel:gpu", 337, 417),
]
TEXT = {
    "en": ("yolo26n on one Intel Core Ultra laptop", "frames per second, end to end, higher is better"),
    "tr": ("Tek bir Intel Core Ultra dizüstünde yolo26n", "saniyedeki kare sayısı, uçtan uca, yüksek olan daha iyi"),
}
THEMES = {
    "light": {"surface": "#ffffff", "ink": "#202124", "muted": "#5f6368", "grid": "#e8eaed"},
    "dark": {"surface": "#17181b", "ink": "#e8eaed", "muted": "#9aa0a6", "grid": "#2d2f33"},
}


def bar(ax, y, width, height, color, surface):
    """A bar with a 4px style rounded data end and a square baseline."""
    radius = min(height / 2, width / 2) if width else 0
    ax.add_patch(
        FancyBboxPatch((0, y), width, height, boxstyle=f"round,pad=0,rounding_size={radius}", color=color, lw=0)
    )
    ax.add_patch(Rectangle((0, y), min(radius, width), height, color=color, lw=0))  # square it off at the baseline


def fps_chart():
    """Grouped bars of Python and Rust FPS per engine and device, one image per theme."""
    if "Inter" in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = "Inter"
    for (theme, t), (lang, (title, xlabel)) in product(THEMES.items(), TEXT.items()):
        fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
        fig.patch.set_facecolor(t["surface"])
        ax.set_facecolor(t["surface"])
        h = 0.34
        for i, (label, py, rs) in enumerate(FPS_ROWS):
            for value, offset, color in ((py, h / 2 + 0.02, GDG[0]), (rs, -h / 2 - 0.02, GDG[1])):
                if value is None:
                    continue
                bar(ax, i + offset - h / 2, value, h, color, t["surface"])
                ax.text(value + 6, i + offset, f"{value}", va="center", ha="left", fontsize=10, color=t["ink"])
        ax.set_yticks(range(len(FPS_ROWS)), [r[0] for r in FPS_ROWS], fontsize=10.5, color=t["ink"])
        ax.set_xlim(0, 470)
        ax.set_ylim(-0.6, len(FPS_ROWS) - 0.4)
        ax.xaxis.grid(True, color=t["grid"], lw=1)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", colors=t["muted"], labelsize=9, length=0)
        ax.tick_params(axis="y", length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_xlabel(xlabel, color=t["muted"], fontsize=9.5)
        fig.suptitle(title, x=0.02, ha="left", color=t["ink"], fontsize=14, weight="bold")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GDG[:2]]
        legend = ax.legend(
            handles, ["Python, Ultralytics YOLO", "Rust, Ultralytics Inference"], loc="lower right", frameon=False
        )
        for text in legend.get_texts():
            text.set_color(t["ink"])
        fig.tight_layout()
        fig.savefig(IMG / f"fps_{lang}_{theme}.png", facecolor=t["surface"])
        plt.close(fig)


def to_gif(frames_dir, gif, fps):
    """Turn numbered PNG frames into a small GIF with an ffmpeg generated palette."""
    palette = (
        f"fps={fps},scale=560:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            f"{frames_dir}/%04d.png",
            "-lavfi",
            palette,
            "-loop",
            "0",
            str(gif),
        ],
        check=True,
    )


def tracking_gif(video="videos/crowd_long.mp4", start=300, seconds=3, fps=10):
    """Three seconds from the middle of the video with fading trails, ID tags and a title bar, as a looping GIF."""
    model = YOLO("yolo26n.pt")
    font = ImageFont.truetype(str(check_font("Arial.ttf")), 18)
    out = Path("runs/visuals/tracking")
    out.mkdir(parents=True, exist_ok=True)
    step = round(30 / fps)
    trails = {}
    for i, result in enumerate(model.track(video, tracker="botsort.yaml", classes=[0], stream=True, verbose=False)):
        if i >= start + seconds * 30:
            break
        frame = result.orig_img.copy()
        live = set()
        if result.boxes.id is not None:
            for (x1, y1, x2, y2), track in zip(result.boxes.xyxy.int().tolist(), result.boxes.id.int().tolist()):
                color = colors(track, True)
                live.add(track)
                trails.setdefault(track, []).append(((x1 + x2) // 2, y2))
                points = trails[track][-25:]
                for k in range(1, len(points)):  # older points get thinner
                    cv2.line(frame, points[k - 1], points[k], color, max(1, k // 6), cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                (tw, th), _ = cv2.getTextSize(f"{track}", cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(
                    frame, f"{track}", (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
                )
        if i < start or i % step:  # the tracker runs from frame 0 so IDs and trails are settled
            continue
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 34), fill=(32, 33, 36))
        for k, c in enumerate(GDG):  # the DevFest four color stripe
            draw.rectangle((k * image.width // 4, 34, (k + 1) * image.width // 4, 39), fill=c)
        title = f"Ultralytics YOLO26 tracking   {len(live)} people in view"
        draw.text((12, 7), title, font=font, fill="white")
        image.save(out / f"{(i - start) // step:04d}.png")
    to_gif(out, IMG / "tracking.gif", fps)


def tasks_gif():
    """Cycle through the seven task outputs from 03_all_tasks.py with a title bar."""
    font = ImageFont.truetype(str(check_font("Arial.ttf")), 26)
    names = ["detect", "segment", "semantic", "depth", "classify", "pose", "obb"]
    width, height = 720, 480
    shots = []
    for k, name in enumerate(names):
        image = Image.open(f"runs/tasks/task_{name}.jpg").convert("RGB")
        image.thumbnail((width, height - 46))
        canvas = Image.new("RGB", (width, height), (32, 33, 36))
        canvas.paste(image, ((width - image.width) // 2, 46 + (height - 46 - image.height) // 2))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 40, width, 44), fill=GDG[k % 4])
        draw.text((14, 6), f"{k + 1}/7  {name}", font=font, fill="white")
        draw.text((width - 14, 6), "one API, seven tasks", font=font, fill=(154, 160, 166), anchor="ra")
        shots.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=200))
    shots[0].save(IMG / "tasks.gif", save_all=True, append_images=shots[1:], duration=1400, loop=0, optimize=True)


def annotate_previews(folder="datasets/auto_annotate_demo"):
    """Draw the polygons written by 07_auto_annotate.py so you can eyeball them."""
    names = YOLO("yolo26n.pt").names
    for label_path in sorted(Path(folder, "labels").glob("*.txt")):
        image = cv2.imread(str(Path(folder, "images", f"{label_path.stem}.jpg")))
        h, w = image.shape[:2]
        overlay = image.copy()
        for line in label_path.read_text().splitlines():
            cls, *coords = line.split()
            polygon = (np.array(coords, dtype=float).reshape(-1, 2) * [w, h]).astype(np.int32)
            color = colors(int(cls), True)
            cv2.fillPoly(overlay, [polygon], color)
            cv2.polylines(image, [polygon], True, color, 2, cv2.LINE_AA)
            x, y = polygon[polygon[:, 1].argmin()]  # tag the top of the mask
            text = names[int(cls)]
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            y = max(int(y), th + 10)
            cv2.rectangle(image, (int(x), y - th - 10), (int(x) + tw + 8, y), color, -1)
            cv2.putText(
                image, text, (int(x) + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
            )
        cv2.imwrite(str(IMG / f"auto_annotate_{label_path.stem}.jpg"), cv2.addWeighted(overlay, 0.4, image, 0.6, 0))


if __name__ == "__main__":
    fps_chart()
    tracking_gif()
    tasks_gif()
    annotate_previews()
    print(f"Saved the charts and GIFs in {IMG}")
