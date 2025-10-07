from typing import Tuple


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two axis-aligned boxes (x, y, w, h)."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_w = max(0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0, min(ay2, by2) - max(ay, by))
    inter = inter_w * inter_h

    area_a = max(0, aw) * max(0, ah)
    area_b = max(0, bw) * max(0, bh)

    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return float(inter) / float(denom)


