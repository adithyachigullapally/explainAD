"""SAM -> consistent part maps for SegAD.

SegAD computes statistics per segment ID, and its head reads fixed columns, so segment l
must mean the same part in every image. SAM's automatic masks are unordered, so each mask
is described by its mean Lab colour and a per-class k-means (fit on good training images)
maps it to one of K part types. Pixels no mask covers get label K.
"""
from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab

SIZE = 256
K = 4              # part types per class, fixed in advance (not tuned)
SAM_RES = 512      # SAM input size; masks are downsampled to SIZE
POINTS = 16        # 16x16 prompt grid: ~4x faster than SAM's default 32x32

_sam = None


def sam_masks(rgb: np.ndarray) -> list[np.ndarray]:
    """(H, W, 3) uint8 -> list of (SIZE, SIZE) bool masks, largest first."""
    global _sam
    if _sam is None:
        from transformers import pipeline
        _sam = pipeline("mask-generation", model="facebook/sam-vit-base", device=0)
    from PIL import Image
    img = Image.fromarray(rgb).resize((SAM_RES, SAM_RES), Image.BILINEAR)
    masks = [np.asarray(m, bool) for m in _sam(img, points_per_crop=POINTS, points_per_batch=16)["masks"]]
    step = SAM_RES // SIZE
    masks = [m[::step, ::step] for m in masks]
    return sorted((m for m in masks if m.any()), key=lambda m: -m.sum())


def colours(rgb: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    """(n, 3) mean Lab colour of each mask, on the SIZE x SIZE image."""
    lab = rgb2lab(rgb)
    return np.array([lab[m].mean(0) for m in masks]).reshape(-1, 3)


def fit_parts(train_rgbs: list[np.ndarray], train_masks: list[list[np.ndarray]], k: int = K):
    from sklearn.cluster import KMeans
    X = np.concatenate([colours(r, m) for r, m in zip(train_rgbs, train_masks)])
    return KMeans(k, n_init=10, random_state=0).fit(X)


def part_map(rgb: np.ndarray, masks: list[np.ndarray], km) -> np.ndarray:
    """(SIZE, SIZE) int32 in [0, K]; small masks painted last so they win overlaps."""
    seg = np.full((SIZE, SIZE), km.n_clusters, np.int32)
    if masks:
        for m, lab in zip(masks, km.predict(colours(rgb, masks))):
            seg[m] = lab
    return seg


if __name__ == "__main__":
    from types import SimpleNamespace
    rgb = np.zeros((SIZE, SIZE, 3), np.uint8)
    rgb[:128] = (200, 30, 30)                       # red top half
    big = np.zeros((SIZE, SIZE), bool); big[:128] = True
    small = np.zeros((SIZE, SIZE), bool); small[10:20, 10:20] = True
    rgb[10:20, 10:20] = (30, 30, 200)               # blue square inside the red half
    km = fit_parts([rgb], [[big, small]], k=2)
    seg = part_map(rgb, [big, small], km)
    assert seg[15, 15] != seg[100, 100], "small mask must overwrite the big one"
    assert (seg[200:] == 2).all(), "uncovered pixels must get label K"
    assert part_map(rgb, [], SimpleNamespace(n_clusters=2)).max() == 2
    print("parts ok")
