import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_split(split_path: str) -> List[str]:
    with open(split_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def draw_title(draw: ImageDraw.ImageDraw, title: str, width: int) -> None:
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    w, h = draw.textlength(title, font=font), 12
    draw.text(((width - w) / 2, 10), title, fill=(0, 0, 0), font=font)


def save_bar_chart(labels: List[str], values: List[int], title: str, out_path: str) -> None:
    width, height = 800, 500
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw_title(draw, title, width)
    margin = 60
    top = 50
    bottom = height - margin
    left = margin
    right = width - margin
    # axes
    draw.line((left, bottom, right, bottom), fill=(0, 0, 0), width=2)
    draw.line((left, bottom, left, top), fill=(0, 0, 0), width=2)
    n = max(1, len(values))
    max_val = max(1, max(values) if values else 1)
    bar_w = (right - left) / (n * 1.5)
    gap = bar_w / 2
    for i, v in enumerate(values):
        x0 = left + i * (bar_w + gap) + gap
        x1 = x0 + bar_w
        h = (v / max_val) * (bottom - top)
        y0 = bottom - h
        color = (76, 120, 168)
        draw.rectangle((x0, y0, x1, bottom - 1), fill=color, outline=(0, 0, 0))
        # label
        lbl = labels[i][:10]
        draw.text((x0, bottom + 5), lbl, fill=(0, 0, 0))
        draw.text((x0, y0 - 12), str(v), fill=(0, 0, 0))
    img.save(out_path)


def save_histogram(values: List[int], bins: int, title: str, out_path: str) -> None:
    width, height = 800, 500
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw_title(draw, title, width)
    margin = 60
    top = 50
    bottom = height - margin
    left = margin
    right = width - margin
    draw.line((left, bottom, right, bottom), fill=(0, 0, 0), width=2)
    draw.line((left, bottom, left, top), fill=(0, 0, 0), width=2)
    if not values:
        values = [0]
    vmin, vmax = min(values), max(values)
    if vmin == vmax:
        vmax = vmin + 1
    bin_edges = [vmin + (vmax - vmin) * i / bins for i in range(bins + 1)]
    counts = [0 for _ in range(bins)]
    for v in values:
        idx = min(bins - 1, int((v - vmin) / (vmax - vmin) * bins))
        counts[idx] += 1
    max_count = max(1, max(counts))
    bar_w = (right - left) / bins
    for i, c in enumerate(counts):
        x0 = left + i * bar_w + 1
        x1 = left + (i + 1) * bar_w - 1
        h = (c / max_count) * (bottom - top)
        y0 = int(bottom - h)
        y0 = max(top, min(y0, bottom - 1))
        color = (229, 87, 86)
        draw.rectangle((x0, y0, x1, bottom - 1), fill=color, outline=(0, 0, 0))
    img.save(out_path)


def gather_stats(annotations_dir: str, names: List[str]) -> Tuple[List[str], List[int], List[int], List[float]]:
    type_counter: Counter = Counter()
    counts_per_image: List[int] = []
    bbox_area_norm: List[float] = []
    for n in names:
        data = load_json(os.path.join(annotations_dir, f"{n}.json"))
        prims = data.get("primitives", [])
        counts_per_image.append(len(prims))
        w = max(1, int(data.get("width", 1)))
        h = max(1, int(data.get("height", 1)))
        img_area = float(w * h)
        for p in prims:
            t = p.get("type", "?")
            type_counter[t] += 1
            x, y, bw, bh = p.get("bbox", [0, 0, 0, 0])
            bbox_area_norm.append(max(0.0, float(bw * bh) / img_area))
    labels = list(type_counter.keys())
    values = [type_counter[k] for k in labels]
    counts_int = [int(c) for c in counts_per_image]
    area_scaled = [int(a * 1000) for a in bbox_area_norm]  # scale for integer hist
    return labels, values, counts_int, area_scaled


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate basic figures without matplotlib.")
    p.add_argument("--annotations-dir", required=True, type=str)
    p.add_argument("--split", required=True, type=str)
    p.add_argument("--out-dir", default=os.path.join("docs", "figs"), type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    names = read_split(args.split)
    ensure_dir(args.out_dir)
    labels, values, counts, area_norm = gather_stats(args.annotations_dir, names)
    save_bar_chart(labels, values, "Primitive Types Distribution", os.path.join(args.out_dir, "primitive_types.png"))
    save_histogram(counts, bins=max(5, min(20, max(counts) + 1 if counts else 5)), title="# Primitives per Image", out_path=os.path.join(args.out_dir, "primitives_per_image.png"))
    save_histogram(area_norm, bins=20, title="BBox Area Distribution (normalized x1000)", out_path=os.path.join(args.out_dir, "bbox_area_hist.png"))
    print(f"Saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()


