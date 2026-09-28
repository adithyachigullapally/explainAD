"""ExplainAD: SAM (parts) -> SegAD (is it defective, where) -> Qwen3-VL (what is wrong, in words).

MVTec 3D-AD RGB images, read from GeoAD's data folder (never written). Every stage caches
per class, so a killed run resumes.

    python pipeline.py fit                    # SAM part maps + PatchCore per class (GPU)
    python pipeline.py eval                   # SegAD heads: 1 segment vs SAM parts (CPU)
    python pipeline.py explain <image.png>    # full pipeline on one image -> results/*.png
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from explainad import head, parts
from explainad.features import segment_stats
from explainad.patchcore import PatchCore

ROOT = Path(__file__).resolve().parent
MV = ROOT.parent / "GeoAD" / "data" / "mvtec3d"
CACHE, RESULTS = ROOT / "cache", ROOT / "results"
CLASSES = sorted(p.name for p in MV.iterdir() if p.is_dir())
SIZE = parts.SIZE
N_KMEANS_IMAGES = 40   # train/good images whose SAM masks define the part types
SEEDS = [333, 576, 725, 823, 831, 902, 226, 598, 874, 589]   # official SegAD seeds
BAD_PARTS = 10         # defective images the head sees per seed (SegAD is supervised)
COOL = 0.5             # GPU rest after each SAM image, x its compute time (0.2 + batch 64 hit 117 W, watchdog kill 09-28)
VLM_ID = "Qwen/Qwen3-VL-2B-Instruct"

GEOMETRIC = {"bent", "cut", "hole", "crack", "open", "thread"}
SURFACE = {"color", "contamination"}


def group_of(t: str) -> str:
    if t == "good":
        return "good"
    return "geometric" if t in GEOMETRIC else "surface" if t in SURFACE else "mixed"


def paths(cls: str, split: str) -> list[Path]:
    return sorted((MV / cls / split).glob("*/rgb/*.png"))


def uid(p: Path) -> str:
    return f"{p.parts[-4]}_{p.parts[-3]}_{p.stem}"   # split_type_stem


def load_rgb(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR))


# ---------------------------------------------------------------- fit
def fit(cls: str) -> None:
    CACHE.mkdir(exist_ok=True)
    todo = paths(cls, "validation") + paths(cls, "test")
    km_f, seg_f = CACHE / f"parts_{cls}.joblib", CACHE / f"segs_{cls}.npy"
    if not seg_f.exists():
        train = paths(cls, "train")
        pick = np.random.default_rng(0).choice(len(train), min(N_KMEANS_IMAGES, len(train)), replace=False)
        rgbs = [load_rgb(train[i]) for i in pick]
        km = parts.fit_parts(rgbs, [_masks(r) for r in rgbs])
        joblib.dump(km, km_f)
        segs = np.stack([parts.part_map(r, _masks(r), km) for r in map(load_rgb, todo)]).astype(np.uint8)
        np.save(seg_f, segs)
        print(f"[sam] {cls}: {len(todo)} part maps", flush=True)
    bank_f, maps_f = CACHE / f"bank_{cls}.pt", CACHE / f"maps_{cls}.npy"
    if not maps_f.exists():
        det = PatchCore(seed=0).fit([load_rgb(p) for p in paths(cls, "train")], batch=4)
        torch.save(det.bank.cpu(), bank_f)
        maps = [m for i in range(0, len(todo), 4)
                for m in det.score(np.stack([load_rgb(p) for p in todo[i:i + 4]]), batch=4)]
        np.save(maps_f, np.stack(maps).astype(np.float16))
        del det
        torch.cuda.empty_cache()
        print(f"[patchcore] {cls}: scored {len(todo)}", flush=True)


def _masks(rgb: np.ndarray) -> list[np.ndarray]:
    t0 = time.perf_counter()
    m = parts.sam_masks(rgb)
    time.sleep(COOL * (time.perf_counter() - t0))
    return m


# ---------------------------------------------------------------- eval
def features(cls: str) -> dict[str, np.ndarray]:
    maps = np.load(CACHE / f"maps_{cls}.npy").astype(np.float32)
    segs = np.load(CACHE / f"segs_{cls}.npy").astype(np.int32)
    one = np.zeros((SIZE, SIZE), np.int32)
    obj = object_mask(cls, segs).astype(np.int32)
    return {"1 segment": np.stack([segment_stats(m, one, 1) for m in maps]),
            "SAM parts": np.stack([segment_stats(m, s, parts.K + 1) for m, s in zip(maps, segs)]),
            "SAM object/bg": np.stack([segment_stats(m, o, 2) for m, o in zip(maps, obj)])}


# Added after the first 4 classes (09-28): colour-clustered parts flip between images (a darker
# defective carrot lands in another cluster), so per-part stats are noisy. Object vs background
# is more stable. Background = the part cluster that fills the image border (objects sit in the
# middle); everything else, including pixels no SAM mask covers, is object. Chosen over an
# "L < 10 is background" rule, which dropped cable glands SAM left uncovered. Known failure:
# tire (black rubber on black) shares the background's cluster, object share 2 %.


def bg_cluster(cls: str) -> int:
    """The part cluster that fills the image border across the class's cached part maps."""
    segs = np.load(CACHE / f"segs_{cls}.npy")
    border = np.concatenate([segs[:, 0], segs[:, -1], segs[:, :, 0], segs[:, :, -1]], axis=1).ravel()
    return int(np.argmax(np.bincount(border, minlength=parts.K + 1)[:parts.K]))


def object_mask(cls: str, segs: np.ndarray) -> np.ndarray:
    """(N, H, W) bool: pixels outside the class's background cluster."""
    return segs != bg_cluster(cls)


def evaluate(cls: str, feats=None) -> list[tuple]:
    todo = paths(cls, "validation") + paths(cls, "test")
    split = np.array([p.parts[-4] for p in todo])
    groups = np.array([group_of(p.parts[-3]) for p in todo])
    y = (groups != "good").astype(int)
    bad = np.flatnonzero((split == "test") & (y == 1))
    X, rows = (feats or features)(cls), []
    for seed in SEEDS:
        tr = split == "validation"
        tr[np.random.default_rng(seed).choice(bad, BAD_PARTS, replace=False)] = True
        for arm, Xa in X.items():
            clf = head.fit(Xa[tr], y[tr], seed=seed)
            p = clf.predict_proba(Xa)[:, 1]
            for g in ("all", "geometric", "surface", "mixed"):
                m = ~tr & ((y == 0) | ((groups == g) if g != "all" else True))
                if (y[m] == 1).any():
                    rows.append((cls, seed, arm, g, 100 * roc_auc_score(y[m], p[m])))
            if seed == SEEDS[0] and arm == "SAM object/bg":   # the demo head (best arm)
                clf.save_model(CACHE / f"head_{cls}.json")
                (CACHE / f"head_{cls}_train.json").write_text(json.dumps([uid(todo[i]) for i in np.flatnonzero(tr)]))
    return rows


def report(classes: list[str], feats=None, name: str = "report") -> None:
    r = pd.DataFrame([row for c in classes for row in evaluate(c, feats)],
                     columns=["cls", "seed", "arm", "group", "auroc"])
    RESULTS.mkdir(exist_ok=True)
    r.to_csv(RESULTS / f"per_seed{name[6:]}.csv", index=False)
    pooled = r.groupby(["arm", "group", "seed"]).auroc.mean().groupby(["arm", "group"]).agg(["mean", "std"])
    table = (pooled["mean"].round(2).astype(str) + " ±" + pooled["std"].round(2).astype(str)).unstack("group")
    per_cls = r[r.group == "all"].groupby(["cls", "arm"]).auroc.mean().unstack("arm").round(1)
    txt = "\n".join([table[["all", "geometric", "surface", "mixed"]].to_string(), "",
                     "AUROC per class (all defects, mean over seeds):", per_cls.to_string()])
    (RESULTS / f"{name}.txt").write_text(txt, encoding="utf-8")
    print(txt)


# ---------------------------------------------------------------- explain
# Demo only (does not touch AUROC). At 0.5 the whole cookie passed and the box was the full
# image (cookie test_crack_000); 0.8 isolates the broken edge. Lower it if boxes clip defects.
BOX_LEVEL = 0.8


def defect_box(amap: np.ndarray, pad: float = 0.35) -> tuple[int, int, int, int]:
    """Box around the connected region of the anomaly peak (map > BOX_LEVEL of its range), padded."""
    from scipy import ndimage
    lab, _ = ndimage.label(amap > amap.min() + BOX_LEVEL * (amap.max() - amap.min()))
    ys, xs = np.nonzero(lab == lab[np.unravel_index(amap.argmax(), amap.shape)])
    h = max(np.ptp(ys), np.ptp(xs), SIZE // 8) * (0.5 + pad)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    return (int(max(cx - h, 0)), int(max(cy - h, 0)), int(min(cx + h, SIZE)), int(min(cy + h, SIZE)))


def describe(image: Image.Image, cls: str) -> str:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(VLM_ID)
    model = AutoModelForImageTextToText.from_pretrained(VLM_ID, dtype=torch.bfloat16).to("cuda").eval()
    q = (f"An inspection system found a defect on this {cls.replace('_', ' ')} inside the red box. "
         "In one sentence, describe the defect in the red box (e.g. crack, hole, cut, bent, "
         "missing piece, contamination, colour spot).")
    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": q}]}]
    inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                   return_dict=True, return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=60, do_sample=False)
    return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def explain(path: Path, threshold: float = 0.5) -> dict:
    from xgboost import XGBClassifier
    cls = path.parts[-5]
    if uid(path) in json.loads((CACHE / f"head_{cls}_train.json").read_text()):
        print(f"warning: {uid(path)} was used to train the demo head; pick another image")
    rgb = load_rgb(path)
    seg = parts.part_map(rgb, parts.sam_masks(rgb), joblib.load(CACHE / f"parts_{cls}.joblib"))
    det = PatchCore(seed=0)
    det.bank = torch.load(CACHE / f"bank_{cls}.pt").to(det.device)
    amap = det.score(rgb[None], batch=1)[0]
    clf = XGBClassifier()
    clf.load_model(CACHE / f"head_{cls}.json")
    obj = (seg != bg_cluster(cls)).astype(np.int32)
    prob = float(clf.predict_proba(segment_stats(amap, obj, 2)[None])[0, 1])
    del det
    torch.cuda.empty_cache()
    res = {"image": str(path), "class": cls, "defect_probability": round(prob, 3), "defective": prob >= threshold}
    box = None
    if res["defective"]:
        box = defect_box(amap)
        # whole image + red box, not a crop: on a crop the VLM loses the context (a missing
        # chunk only looks wrong next to the intact shape) and answered "no visible defects"
        from PIL import ImageDraw
        img = Image.fromarray(rgb).resize((448, 448), Image.BICUBIC)
        s = 448 / SIZE
        ImageDraw.Draw(img).rectangle([v * s for v in box], outline=(255, 0, 0), width=4)
        res["description"] = describe(img, cls)
    _panel(rgb, seg, amap, box, res, RESULTS / f"explain_{cls}_{uid(path)}.png")
    return res


def _panel(rgb, seg, amap, box, res, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.6))
    ax[0].imshow(rgb); ax[0].set_title("input")
    ax[1].imshow(seg, cmap="tab10", vmin=0, vmax=9); ax[1].set_title("SAM parts")
    ax[2].imshow(rgb); ax[2].imshow(amap, cmap="jet", alpha=0.45); ax[2].set_title("SegAD anomaly map")
    if box:
        ax[2].add_patch(Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False, ec="white", lw=2))
    for a in ax:
        a.axis("off")
    verdict = "DEFECT" if res["defective"] else "OK"
    fig.suptitle(f"{verdict}  (p = {res['defect_probability']:.2f})   {res.get('description', '')}", wrap=True)
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    res["panel"] = str(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "eval", "explain"])
    ap.add_argument("image", nargs="?", type=Path)
    ap.add_argument("--classes", nargs="+", default=CLASSES)
    a = ap.parse_args()
    if a.stage == "fit":
        for c in a.classes:
            fit(c)
    elif a.stage == "eval":
        report(a.classes)
    else:
        print(json.dumps(explain(a.image), indent=2))


if __name__ == "__main__":
    assert group_of("hole") == "geometric" and group_of("combined") == "mixed"
    assert uid(Path("x/bagel/test/crack/rgb/000.png")) == "test_crack_000"
    b = np.zeros((SIZE, SIZE)); b[100:110, 50:60] = 1
    x0, y0, x1, y1 = defect_box(b)
    assert x0 <= 50 and y0 <= 100 and x1 >= 60 and y1 >= 110, (x0, y0, x1, y1)
    main()
