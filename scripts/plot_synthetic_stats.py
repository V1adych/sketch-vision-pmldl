import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_primitive_types(annotations_dir: str, names: List[str], out_dir: str) -> None:
    counter: Counter = Counter()
    for n in names:
        data = load_json(os.path.join(annotations_dir, f"{n}.json"))
        for p in data.get("primitives", []):
            counter[p.get("type", "?")] += 1
    labels = list(counter.keys())
    values = [counter[k] for k in labels]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color="#4C78A8")
    plt.title("Primitive Types Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    ensure_dir(out_dir)
    plt.savefig(os.path.join(out_dir, "primitive_types.png"), dpi=200)
    plt.close()


def plot_primitives_per_image(annotations_dir: str, names: List[str], out_dir: str) -> None:
    counts: List[int] = []
    for n in names:
        data = load_json(os.path.join(annotations_dir, f"{n}.json"))
        counts.append(len(data.get("primitives", [])))
    plt.figure(figsize=(6, 4))
    plt.hist(counts, bins=range(0, max(counts) + 2), color="#72B7B2", edgecolor="black")
    plt.title("Primitives per Image")
    plt.xlabel("# primitives")
    plt.ylabel("# images")
    plt.tight_layout()
    ensure_dir(out_dir)
    plt.savefig(os.path.join(out_dir, "primitives_per_image.png"), dpi=200)
    plt.close()


def plot_bbox_area_hist(annotations_dir: str, names: List[str], out_dir: str) -> None:
    areas: List[float] = []
    for n in names:
        data = load_json(os.path.join(annotations_dir, f"{n}.json"))
        w = max(1, int(data.get("width", 1)))
        h = max(1, int(data.get("height", 1)))
        img_area = float(w * h)
        for p in data.get("primitives", []):
            x, y, bw, bh = p.get("bbox", [0, 0, 0, 0])
            areas.append(max(0.0, float(bw * bh) / img_area))
    if not areas:
        areas = [0.0]
    plt.figure(figsize=(6, 4))
    plt.hist(areas, bins=20, color="#E45756", edgecolor="black")
    plt.title("BBox Area Distribution (normalized)")
    plt.xlabel("bbox_area / image_area")
    plt.ylabel("count")
    plt.tight_layout()
    ensure_dir(out_dir)
    plt.savefig(os.path.join(out_dir, "bbox_area_hist.png"), dpi=200)
    plt.close()


def read_split(split_path: str) -> List[str]:
    with open(split_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot charts from synthetic annotations.")
    p.add_argument("--annotations-dir", required=True, type=str)
    p.add_argument("--split", required=True, type=str, help="Path to split file (train/val/test .txt)")
    p.add_argument("--out-dir", default=os.path.join("docs", "figs"), type=str, help="Output figs directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    names = read_split(args.split)
    ensure_dir(args.out_dir)
    plot_primitive_types(args.annotations_dir, names, args.out_dir)
    plot_primitives_per_image(args.annotations_dir, names, args.out_dir)
    plot_bbox_area_hist(args.annotations_dir, names, args.out_dir)
    print(f"Saved charts to {args.out_dir}")


if __name__ == "__main__":
    main()


