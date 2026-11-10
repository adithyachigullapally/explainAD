"""How good is the VLM explanation? Pre-registered 2026-10-07 before any number.

Sample: 8 random defective test images per (class, defect type), seed 0, excluding images the demo head
trained on. The box comes from the cached DINO map (same rule as the demo; WRN50 until 11-10: 37.7 % localised),
no detection gate.
  localised = box centre lies inside the ground-truth defect mask.
  type match = description contains a keyword for the true type (table below; 'combined' = any keyword).
Run: python vlm_eval.py -> results/vlm_eval.txt
"""
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import pipeline as pl

KEYS = {"crack": ["crack", "fracture", "split", "broken", "break"], "hole": ["hole", "puncture", "missing"],
        "cut": ["cut", "slit", "slice", "tear", "scratch", "gash"], "bent": ["bent", "bend", "deform", "warp", "curv"],
        "contamination": ["contaminat", "foreign", "stain", "dirt", "debris", "particle", "spot"],
        "color": ["color", "colour", "discolor", "stain", "spot"], "open": ["open", "unravel", "fray", "loose", "split"],
        "thread": ["thread", "damage", "stripped"]}
KEYS["combined"] = sorted({k for v in KEYS.values() for k in v})


def match(text: str, kind: str) -> bool:
    return any(k in text.lower() for k in KEYS[kind])


def first_type(text: str) -> str:
    """The defect type whose keyword comes first in a description (demo label; right ~41 % of the time)."""
    t = text.lower()
    hits = {k: min(t.find(w) for w in ws if w in t) for k, ws in KEYS.items() if k != "combined" and match(t, k)}
    return min(hits, key=hits.get) if hits else "type unclear"


if __name__ == "__main__":
    assert match("a small crack", "crack") and not match("a small crack", "hole") and match("a hole", "combined")
    assert first_type("A hole next to a crack") == "hole" and first_type("looks fine") == "type unclear"
    rows = []
    for cls in pl.CLASSES:
        todo = pl.paths(cls, "validation") + pl.paths(cls, "test")
        maps = np.load(pl.detectors(cls)[1][3]).astype(np.float32)   # DINO, as in pl.inspect
        used = set(json.loads((pl.CACHE / f"head_{cls}_train.json").read_text()))
        by_kind = {}
        for i, p in enumerate(todo):
            if p.parts[-4] == "test" and p.parts[-3] != "good" and pl.uid(p) not in used:
                by_kind.setdefault(p.parts[-3], []).append(i)
        for kind, idx in sorted(by_kind.items()):
            for i in np.random.default_rng(0).choice(idx, min(8, len(idx)), replace=False):
                p = todo[i]
                x0, y0, x1, y1 = box = pl.defect_box(maps[i])
                gt = np.asarray(Image.open(p.parent.parent / "gt" / p.name).convert("L").resize((pl.SIZE, pl.SIZE))) > 0
                hit = bool(gt[(y0 + y1) // 2, (x0 + x1) // 2])
                img = Image.fromarray(pl.load_rgb(p)).resize((448, 448), Image.BICUBIC)
                ImageDraw.Draw(img).rectangle([v * 448 / pl.SIZE for v in box], outline=(255, 0, 0), width=4)
                text = pl.describe(img, cls, free=False)
                rows.append((cls, kind, hit, match(text, kind), text))
        print(cls, len(rows), flush=True)
    r = pd.DataFrame(rows, columns=["cls", "kind", "localised", "type_match", "text"])
    r.to_csv(pl.RESULTS / "vlm_eval.csv", index=False)
    out = [f"n = {len(r)}  box on defect: {100 * r.localised.mean():.1f} %  type named right: {100 * r.type_match.mean():.1f} %",
           f"type right when box on defect: {100 * r[r.localised].type_match.mean():.1f} %", "",
           r.groupby("kind")[["localised", "type_match"]].mean().mul(100).round(1).to_string()]
    (pl.RESULTS / "vlm_eval.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
