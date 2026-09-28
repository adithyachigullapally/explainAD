"""Improvement ablation, pre-registered in the Obsidian note ExplainAD (2026-09-28 ~15:10).

Four arms, run once, all reported. Segments = SAM object/background in every arm.
  A  full image,       PatchCore-WRN50                     (= the 82.4 result)
  B  SAM object crop,  PatchCore-WRN50
  C  full image,       PatchCore-WRN50 + PatchCore-DINO (ICCV 2021)
  D  SAM object crop,  PatchCore-WRN50 + PatchCore-DINO

    python improve.py fit     # SAM on all train images, crop boxes, 3 new map streams (GPU, resumable)
    python improve.py eval    # results/report_improve.txt
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

import pipeline as P
from explainad import parts
from explainad.features import segment_stats
from explainad.patchcore import DinoCore, PatchCore

SIZE, CACHE = P.SIZE, P.CACHE
MARGIN, MIN_SIDE = 1.15, 32   # crop box: square, side x 1.15, at least 32 px (256-px coords)
NO_ZOOM_BELOW = 0.04          # classes whose mean object share is below this keep the full image (tire, 2 %)


def crop_box(obj: np.ndarray) -> tuple[int, int, int, int]:
    """Square box (256-px coords) around the largest connected object component."""
    lab, n = ndimage.label(obj)
    if n == 0:
        return 0, 0, SIZE, SIZE
    ys, xs = np.nonzero(lab == np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    side = int(round(min(max(max(np.ptp(ys), np.ptp(xs)) + 1, MIN_SIDE) * MARGIN, SIZE)))
    x0 = int(np.clip(round((xs.min() + xs.max()) / 2 - side / 2), 0, SIZE - side))
    y0 = int(np.clip(round((ys.min() + ys.max()) / 2 - side / 2), 0, SIZE - side))
    return x0, y0, x0 + side, y0 + side


def load_zoom(p, box) -> np.ndarray:
    """Crop from the ORIGINAL-resolution image (that is where the extra detail comes from)."""
    im = Image.open(p).convert("RGB")
    sx, sy = im.size[0] / SIZE, im.size[1] / SIZE
    x0, y0, x1, y1 = box
    return np.asarray(im.crop((x0 * sx, y0 * sy, x1 * sx, y1 * sy)).resize((SIZE, SIZE), Image.BILINEAR))


def crop_seg(seg: np.ndarray, box) -> np.ndarray:
    return np.asarray(Image.fromarray(seg.astype(np.uint8)).crop(box).resize((SIZE, SIZE), Image.NEAREST))


def boxes(cls: str) -> dict[str, np.ndarray]:
    """Crop boxes for train and val+test images; full image for no-zoom classes."""
    f = CACHE / f"boxes_{cls}.npz"
    if f.exists():
        return dict(np.load(f))
    bg = P.bg_cluster(cls)
    out = {}
    for k, seg_f in (("train", f"segs_train_{cls}.npy"), ("todo", f"segs_{cls}.npy")):
        obj = np.load(CACHE / seg_f) != bg
        out[k] = np.array([crop_box(o) for o in obj])
    if (np.load(CACHE / f"segs_{cls}.npy") != bg).mean() < NO_ZOOM_BELOW:
        out = {k: np.tile([0, 0, SIZE, SIZE], (len(v), 1)) for k, v in out.items()}
    np.savez(f, **out)
    return out


def fit(cls: str) -> None:
    train, todo = P.paths(cls, "train"), P.paths(cls, "validation") + P.paths(cls, "test")
    seg_f = CACHE / f"segs_train_{cls}.npy"
    if not seg_f.exists():
        km = joblib.load(CACHE / f"parts_{cls}.joblib")
        segs = np.stack([parts.part_map(r, P._masks(r), km) for r in map(P.load_rgb, train)]).astype(np.uint8)
        np.save(seg_f, segs)
        print(f"[sam-train] {cls}: {len(train)} part maps", flush=True)
    b = boxes(cls)
    full = lambda ps, _: [P.load_rgb(p) for p in ps]                        # noqa: E731
    zoom = lambda ps, bx: [load_zoom(p, x) for p, x in zip(ps, bx)]         # noqa: E731
    for name, Det, load in (("wrn_zoom", PatchCore, zoom), ("dino_full", DinoCore, full), ("dino_zoom", DinoCore, zoom)):
        f = CACHE / f"maps_{name}_{cls}.npy"
        if f.exists():
            continue
        det = Det(seed=0).fit(load(train, b["train"]), batch=4)
        maps = [m for i in range(0, len(todo), 4)
                for m in det.score(np.stack(load(todo[i:i + 4], b["todo"][i:i + 4])), batch=4)]
        np.save(f, np.stack(maps).astype(np.float16))
        del det
        torch.cuda.empty_cache()
        print(f"[{name}] {cls}: scored {len(todo)}", flush=True)


def features(cls: str) -> dict[str, np.ndarray]:
    segs = np.load(CACHE / f"segs_{cls}.npy")
    bg, bx = P.bg_cluster(cls), boxes(cls)["todo"]
    obj = (segs != bg).astype(np.int32)
    obj_z = np.stack([crop_seg(s, x) != bg for s, x in zip(segs, bx)]).astype(np.int32)
    stats = lambda name, o: np.stack([segment_stats(m, s, 2) for m, s in zip(  # noqa: E731
        np.load(CACHE / f"maps_{name}_{cls}.npy" if name else CACHE / f"maps_{cls}.npy").astype(np.float32), o)])
    wrn, wrn_z = stats("", obj), stats("wrn_zoom", obj_z)
    return {"A current": wrn,
            "B + zoom": wrn_z,
            "C + DINO": np.hstack([wrn, stats("dino_full", obj)]),
            "D + zoom + DINO": np.hstack([wrn_z, stats("dino_zoom", obj_z)])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "eval"])
    ap.add_argument("--classes", nargs="+", default=P.CLASSES)
    a = ap.parse_args()
    if a.stage == "fit":
        for c in a.classes:
            fit(c)
    else:
        P.report(a.classes, features, name="report_improve")


if __name__ == "__main__":
    o = np.zeros((SIZE, SIZE), bool); o[100:120, 40:200] = True; o[5, 5] = True   # object + a speck
    x0, y0, x1, y1 = crop_box(o)
    assert x1 - x0 == y1 - y0 and x0 <= 40 and x1 >= 200 and y0 <= 100 and y1 >= 120, (x0, y0, x1, y1)
    assert x0 > 5 or y0 > 5, "speck must not enlarge the box"
    assert crop_box(np.zeros((SIZE, SIZE), bool)) == (0, 0, SIZE, SIZE)
    assert crop_seg(np.eye(SIZE, dtype=np.uint8), (0, 0, 128, 128)).shape == (SIZE, SIZE)
    main()
