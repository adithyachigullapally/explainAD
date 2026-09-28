"""Round 2 ablation, pre-registered in the Obsidian note ExplainAD (2026-09-28 ~17:20). Last round.

All arms: SAM object/background segments, same SegAD head, seeds and splits.
  C  WRN50 + DINO         at 256 px   (= v2, 85.2)
  E  WRN50 + DINO         at 384 px
  F  WRN50 + DINO + CLIP  at 256 px
  G  WRN50 + DINO + CLIP  at 384 px

    python improve2.py fit    # 4 new map streams per class (GPU, resumable)
    python improve2.py eval   # results/report_improve2.txt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

import pipeline as P
from explainad.features import segment_stats
from explainad.patchcore import ClipCore, DinoCore, PatchCore

CACHE, HI = P.CACHE, 384
STREAMS = {"wrn_384": (PatchCore, HI), "dino_384": (DinoCore, HI),
           "clip_256": (ClipCore, P.SIZE), "clip_384": (ClipCore, HI)}


def load(p, size: int) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR))


def fit(cls: str) -> None:
    train, todo = P.paths(cls, "train"), P.paths(cls, "validation") + P.paths(cls, "test")
    for name, (Det, size) in STREAMS.items():
        f = CACHE / f"maps_{name}_{cls}.npy"
        if f.exists():
            continue
        det = Det(seed=0).fit([load(p, size) for p in train], batch=2)
        # maps come back at 256 (patchcore.SIZE), so the SAM masks line up at every input size
        maps = [m for i in range(0, len(todo), 2)
                for m in det.score(np.stack([load(p, size) for p in todo[i:i + 2]]), batch=2)]
        np.save(f, np.stack(maps).astype(np.float16))
        del det
        torch.cuda.empty_cache()
        print(f"[{name}] {cls}: scored {len(todo)}", flush=True)


def features(cls: str) -> dict[str, np.ndarray]:
    obj = P.object_mask(cls, np.load(CACHE / f"segs_{cls}.npy")).astype(np.int32)
    s = {n: np.stack([segment_stats(m, o, 2) for m, o in zip(np.load(CACHE / f).astype(np.float32), obj)])
         for n, f in (("wrn", f"maps_{cls}.npy"), ("dino", f"maps_dino_full_{cls}.npy"),
                      *((k, f"maps_{k}_{cls}.npy") for k in STREAMS))}
    return {"C v2 (256)": np.hstack([s["wrn"], s["dino"]]),
            "E + 384 px": np.hstack([s["wrn_384"], s["dino_384"]]),
            "F + CLIP": np.hstack([s["wrn"], s["dino"], s["clip_256"]]),
            "G + 384 px + CLIP": np.hstack([s["wrn_384"], s["dino_384"], s["clip_384"]])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "eval"])
    ap.add_argument("--classes", nargs="+", default=P.CLASSES)
    a = ap.parse_args()
    if a.stage == "fit":
        for c in a.classes:
            fit(c)
    else:
        P.report(a.classes, features, name="report_improve2")


if __name__ == "__main__":
    assert load(P.paths("cookie", "test")[0], HI).shape == (HI, HI, 3)
    main()
