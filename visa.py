"""Second dataset, camera mode only: VisA (Zou et al., ECCV 2022; 12 factory-style classes, colour images, no 3D scans).
Pre-registered 2026-11-10 before any code was run or any number seen; run once, every arm reported.

Question (RQ2): do the camera-mode gains measured on MVTec 3D-AD (SAM object/background, then DINO, then CLIP as extra
PatchCore detectors) hold on a second dataset?

Protocol = the SegAD paper's own VisA benchmark (Baitieva et al., CVPR 2024, Sec. 5.4): VisA's `1cls` split; 10 % of
each class's good training images (random, seed 0) are held out to train the SegAD head and are not in the PatchCore
memory banks; 10 random defective test images are moved to head training, for the same 10 seeds; evaluation on the
remaining test images. Everything else is the frozen pipeline: same SAM settings, colour k-means (K = 4) on 40 good
training images, background = the cluster on the image border, PatchCore on WRN50, DINO and CLIP at 256 px, same head.
Arms: 1 segment (WRN50) -> SAM object/bg -> + DINO -> + CLIP (the MVTec 3D-AD ladder, without the depth arms).
Not done: the paper's extra static component maps for pcb1-4 (we use plain object/background on every class).
The paper's VisA numbers (RD4AD / EfficientAD based, 96.1-98.4) use other base detectors: not comparable row by row.
The object masks are rendered (`masks`) and looked at before any score.

    python visa.py prepare   # data/visa_raw (extracted VisA_20220922.tar) -> data/visa, the pipeline's folder layout (hard links)
    python visa.py fit       # SAM part maps + the three PatchCore detectors per class (GPU, resumable)
    python visa.py masks     # results/visa_masks.png: SAM object mask on one test image per class (label-free check)
    python visa.py eval      # results/report_visa.txt, per_seed_visa.csv, operating_point_visa.txt
    python visa.py calibrate # per-class demo threshold -> cache_visa/thresholds.json (CPU; does not touch AUROC)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import pipeline as P
from explainad.features import segment_stats

RAW = P.ROOT / "data" / "visa_raw"
P.MV, P.CACHE = P.ROOT / "data" / "visa", P.ROOT / "cache_visa"   # the pipeline reads these two at call time
CLASSES = sorted(p.name for p in P.MV.iterdir() if p.is_dir()) if P.MV.is_dir() else []
VAL_SHARE = 0.1


def prepare() -> None:
    df = pd.read_csv(RAW / "split_csv" / "1cls.csv")
    for cls, g in df.groupby("object"):
        train = np.sort(g[g.split == "train"].image.to_numpy())
        val = set(np.random.default_rng(0).choice(train, round(VAL_SHARE * len(train)), replace=False))
        assert set(g[g.split == "train"].label) == {"normal"}, f"{cls}: defective image in the training split"
        for r in g.itertuples():
            split = "validation" if r.image in val else r.split
            dst = P.MV / cls / split / ("good" if r.label == "normal" else "bad") / "rgb" / Path(r.image).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                os.link(RAW / r.image, dst)   # hard link: no second copy of the 1.9 GB
        n = {s: len(P.paths(cls, s)) for s in ("train", "validation", "test")}
        assert sum(n.values()) == len(g) and n["validation"] == len(val), (cls, n, len(g))
        print(cls, n, flush=True)


def features(cls: str) -> dict[str, np.ndarray]:
    obj = P.object_mask(cls, np.load(P.CACHE / f"segs_{cls}.npy")).astype(np.int32)
    maps = [np.load(d[3]).astype(np.float32) for d in P.detectors(cls)[:3]]
    s = [np.stack([segment_stats(m, o, 2) for m, o in zip(mp, obj)]) for mp in maps]
    return {"1 segment": np.stack([segment_stats(m, np.zeros_like(obj[0]), 1) for m in maps[0]]),
            "SAM object/bg": s[0],
            "SAM object/bg + DINO": np.hstack(s[:2]),
            "SAM object/bg + DINO + CLIP": np.hstack(s)}


def masks(classes: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, len(classes), figsize=(2.2 * len(classes), 4.6), squeeze=False)
    for j, cls in enumerate(classes):
        todo = P.paths(cls, "validation") + P.paths(cls, "test")
        obj = P.object_mask(cls, np.load(P.CACHE / f"segs_{cls}.npy"))
        i = len(todo) - 1   # the last test image
        ax[0, j].imshow(P.load_rgb(todo[i])); ax[0, j].set_title(cls, fontsize=9)
        ax[1, j].imshow(obj[i], cmap="gray"); ax[1, j].set_title(f"object {100 * obj.mean():.0f} %", fontsize=9)
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout()
    fig.savefig(P.RESULTS / "visa_masks.png", dpi=110)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "fit", "masks", "eval", "calibrate"])
    ap.add_argument("--classes", nargs="+", default=CLASSES)
    a = ap.parse_args()
    if a.stage == "prepare":
        prepare()
    elif a.stage == "fit":
        for c in a.classes:
            P.fit(c, dets=3)
    elif a.stage == "masks":
        masks(a.classes)
    elif a.stage == "calibrate":   # demo thresholds only, same rule as pipeline.py calibrate; test images unused
        P.features = features      # P.calibrate looks features up at call time
        out = P.CACHE / "thresholds.json"
        t = {c: P.calibrate(c) for c in a.classes}
        out.write_text(json.dumps(t, indent=1))
        print(json.dumps(t))
    else:
        P.report(a.classes, features, name="report_visa")
