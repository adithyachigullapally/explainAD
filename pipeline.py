"""ExplainAD: SAM (object/background) -> SegAD on PatchCore maps from WRN50, DINO and CLIP (is it defective,
where) -> Qwen3-VL (what is wrong, in words).

MVTec 3D-AD images, read from data/mvtec3d (never written). Every stage caches
per class, so a killed run resumes.

    python pipeline.py fit                    # SAM part maps + PatchCore (WRN50, DINO, CLIP) per class (GPU)
    python pipeline.py eval                   # SegAD heads for every setup -> results/report.txt (CPU)
    python pipeline.py calibrate              # per-class demo threshold -> cache/thresholds.json (CPU)
    python pipeline.py explain <image.png> [--3d]   # full pipeline on one image -> results/*.png
    streamlit run app.py                      # the same, as a web demo
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
from explainad.patchcore import ClipCore, DinoCore, PatchCore

ROOT = Path(__file__).resolve().parent
MV = ROOT / "data" / "mvtec3d"
CACHE, RESULTS = ROOT / "cache", ROOT / "results"
CLASSES = sorted(p.name for p in MV.iterdir() if p.is_dir()) if MV.is_dir() else []   # empty: dataset not downloaded
SIZE = parts.SIZE
N_KMEANS_IMAGES = 40   # train/good images whose SAM masks define the part types
SEEDS = [333, 576, 725, 823, 831, 902, 226, 598, 874, 589]   # official SegAD seeds
BAD_PARTS = 10         # defective images the head sees per seed (SegAD is supervised)
COOL = 1.0             # GPU rest after each SAM image, x its compute time. 0.2+batch 64 hit 117 W (13:07); 0.5 hit 136 W twice in a row (15:09)
VLM_ID = "Qwen/Qwen3-VL-2B-Instruct"

GEOMETRIC = {"bent", "cut", "hole", "crack", "open", "thread"}
SURFACE = {"color", "contamination"}


def group_of(t: str) -> str:
    if t == "good":
        return "good"
    return "geometric" if t in GEOMETRIC else "surface" if t in SURFACE else "mixed"


def paths(cls: str, split: str) -> list[Path]:
    return sorted((MV / cls / split).glob("*/rgb/*.*"))   # .png in MVTec 3D-AD, .JPG in VisA (visa.py)


def uid(p: Path) -> str:
    return f"{p.parts[-4]}_{p.parts[-3]}_{p.stem}"   # split_type_stem


def load_rgb(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR))


def load_depth(p: Path) -> np.ndarray:
    """Real 3D-sensor depth for the RGB image at p (optional 3D mode): z channel of the xyz scan,
    sensor holes filled with the nearest valid reading, near = high, min-max to [0, 1], SIZE px."""
    import tifffile
    from scipy import ndimage
    z = tifffile.imread(p.parent.parent / "xyz" / f"{p.stem}.tiff")[..., 2].astype(np.float32)
    valid = z > 0
    assert valid.any(), f"no valid depth in {p}"
    z = -z[tuple(ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True))]
    z = np.asarray(Image.fromarray(z).resize((SIZE, SIZE), Image.BILINEAR))
    return ((z - z.min()) / (z.max() - z.min() + 1e-8)).astype(np.float32)


# ---------------------------------------------------------------- fit
def fit(cls: str, dets: int = 4) -> None:
    """dets = 3 skips the depth detector (visa.py: a dataset without 3D scans)."""
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
    for name, Det, bank_f, maps_f, batch, load in detectors(cls)[:dets]:
        if bank_f.exists() and maps_f.exists():
            continue
        det = Det(seed=0).fit([load(p) for p in paths(cls, "train")], batch=batch)
        torch.save(det.bank.cpu(), bank_f)
        maps = [m for i in range(0, len(todo), batch)
                for m in det.score(np.stack([load(p) for p in todo[i:i + batch]]), batch=batch)]
        np.save(maps_f, np.stack(maps).astype(np.float16))
        del det
        torch.cuda.empty_cache()
        print(f"[{name}] {cls}: scored {len(todo)}", flush=True)


# The three PatchCore detectors SegAD combines. DINO (ICCV 2021) added +2.8 AUROC and CLIP
# (ICML 2021) another +2.3 in the pre-registered ablations (improve.py, improve2.py). CLIP keeps
# batch 2, as in its ablation: batch size changes PatchCore's random memory-bank sample.
# The fourth, optional detector is the 3D mode: PatchCore-WRN50 on real sensor depth. Camera-only
# mode uses the first three; with a 3D scanner, all four.
def detectors(cls: str):
    return [("wrn", PatchCore, CACHE / f"bank_{cls}.pt", CACHE / f"maps_{cls}.npy", 4, load_rgb),
            ("dino", DinoCore, CACHE / f"bank_dino_{cls}.pt", CACHE / f"maps_dino_full_{cls}.npy", 4, load_rgb),
            ("clip", ClipCore, CACHE / f"bank_clip_{cls}.pt", CACHE / f"maps_clip_256_{cls}.npy", 2, load_rgb),
            ("depth", PatchCore, CACHE / f"bank_depth_{cls}.pt", CACHE / f"maps_depth_{cls}.npy", 4, load_depth)]


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
    single = np.stack([segment_stats(m, one, 1) for m in maps])
    depth = np.load(detectors(cls)[3][3]).astype(np.float32)
    return {"1 segment": single,
            # baseline: plain SegAD + real depth, no SAM
            "1 segment + real depth": np.hstack([single, np.stack([segment_stats(m, one, 1) for m in depth])]),
            "SAM parts": np.stack([segment_stats(m, s, parts.K + 1) for m, s in zip(maps, segs)]),
            "SAM object/bg": np.stack([segment_stats(m, o, 2) for m, o in zip(maps, obj)]),
            **{arm: np.hstack([np.stack([segment_stats(m, o, 2) for m, o in zip(
                np.load(d[3]).astype(np.float32), obj)]) for d in detectors(cls)[:n]])
               for arm, n in (("SAM object/bg + DINO", 2), ("SAM object/bg + DINO + CLIP", 3),
                              ("with 3D scanner: + real depth", 4))}}


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


OPS: list = []   # (cls, seed, arm, recall %, false-alarm %) at threshold 0.5, held-out images
DEMO_HEADS = {"SAM object/bg + DINO + CLIP": "head", "with 3D scanner: + real depth": "head3d"}


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
            if arm in DEMO_HEADS:
                m = ~tr
                OPS.append((cls, seed, arm, 100 * (p[m & (y == 1)] >= .5).mean(), 100 * (p[m & (y == 0)] >= .5).mean()))
            for g in ("all", "geometric", "surface", "mixed"):
                m = ~tr & ((y == 0) | ((groups == g) if g != "all" else True))
                if (y[m] == 1).any():
                    rows.append((cls, seed, arm, g, 100 * roc_auc_score(y[m], p[m])))
            if seed == SEEDS[0] and arm in DEMO_HEADS:   # the demo heads: camera mode, 3D mode
                clf.save_model(CACHE / f"{DEMO_HEADS[arm]}_{cls}.json")
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
    groups = [g for g in ("all", "geometric", "surface", "mixed") if g in table]   # VisA has no defect-type folders
    txt = "\n".join([table[groups].to_string(), "",
                     "AUROC per class (all defects, mean over seeds):", per_cls.to_string()])
    (RESULTS / f"{name}.txt").write_text(txt, encoding="utf-8")
    print(txt)
    if not OPS:   # improve.py / improve2.py arms are not demo heads: keep the existing file
        return
    op = pd.DataFrame(OPS, columns=["cls", "seed", "arm", "recall", "false_alarm"])
    op = op.groupby(["arm", "cls"])[["recall", "false_alarm"]].mean().round(1)
    (RESULTS / f"operating_point{name[6:]}.txt").write_text("Threshold 0.5, held-out images, mean over seeds (%):\n"
                                                 + op.to_string(), encoding="utf-8")


# ---------------------------------------------------------------- calibrate
# Per-class defect threshold for the demo (does not touch AUROC). Rule fixed 2026-10-13 before
# looking at its test results: leave-one-out over the demo head's own training images (validation
# good + its 10 defective), then the slider value (0.05 steps) with the best balanced accuracy on
# those left-out probabilities; ties go to the value nearest 0.5. Test images are never used.
THRESHOLDS = CACHE / "thresholds.json"
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def pick_threshold(y: np.ndarray, p: np.ndarray) -> float:
    bal = np.array([((p[y == 1] >= t).mean() + (p[y == 0] < t).mean()) / 2 for t in GRID])
    best = GRID[np.isclose(bal, bal.max())]
    return float(best[np.abs(best - 0.5).argmin()])


def calibrate(cls: str) -> dict[str, float]:
    todo = paths(cls, "validation") + paths(cls, "test")
    used = set(json.loads((CACHE / f"head_{cls}_train.json").read_text()))
    tr = np.flatnonzero([uid(p) in used for p in todo])
    y = np.array([todo[i].parts[-3] != "good" for i in tr]).astype(int)
    X, out = features(cls), {}
    for arm, name in DEMO_HEADS.items():
        if arm not in X:   # VisA (visa.py): no depth, camera head only
            continue
        Xa = X[arm][tr]
        p = np.array([head.fit(np.delete(Xa, i, 0), np.delete(y, i), seed=SEEDS[0]).predict_proba(Xa[i:i + 1])[0, 1]
                      for i in range(len(y))])
        out["3D" if name == "head3d" else "camera"] = pick_threshold(y, p)
    return out


def demo_threshold(cls: str, mode: str = "camera") -> float:
    """The class's calibrated threshold (python pipeline.py calibrate), 0.5 if it was never calibrated."""
    f = CACHE / "thresholds.json"   # at call time: the demo app switches CACHE between datasets
    return json.loads(f.read_text()).get(cls, {}).get(mode, 0.5) if f.exists() else 0.5


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


_VLM: dict = {}


def describe(image: Image.Image, cls: str, free: bool = True) -> str:
    """One-sentence description. free=False keeps the VLM on the GPU for batch use (vlm_eval.py).
    Tried 11-10: adding a defect-free image of the class as a reference; type named right 34.6 % vs 41.0 % (78 images), dropped."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if "m" not in _VLM:
        _VLM["p"] = AutoProcessor.from_pretrained(VLM_ID)
        _VLM["m"] = AutoModelForImageTextToText.from_pretrained(VLM_ID, dtype=torch.bfloat16).to("cuda").eval()
    proc, model = _VLM["p"], _VLM["m"]
    q = (f"An inspection system found a defect on this {cls.replace('_', ' ')} inside the red box. "
         "In one sentence, describe the defect in the red box (e.g. crack, hole, cut, bent, "
         "missing piece, contamination, colour spot).")
    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": q}]}]
    inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                   return_dict=True, return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=60, do_sample=False)
    text = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    if free:
        _VLM.clear()
    del model, out, inp
    torch.cuda.empty_cache()
    return text


_DETS: dict = {}   # backbones stay loaded between calls (demo app); only the per-class bank is swapped


def inspect(rgb: np.ndarray, cls: str, depth: np.ndarray | None = None, threshold: float = 0.5,
            out: Path | None = None) -> dict:
    """Full pipeline on one (SIZE, SIZE, 3) image of class cls. depth given = 3D mode (scanner), else camera mode."""
    from xgboost import XGBClassifier
    seg = parts.part_map(rgb, parts.sam_masks(rgb), joblib.load(CACHE / f"parts_{cls}.joblib"))
    maps = []
    for name, Det, bank_f, _, _, _ in detectors(cls)[:4 if depth is not None else 3]:
        det = _DETS.setdefault(name, Det(seed=0))
        det.bank = torch.load(bank_f).to(det.device)
        maps.append(det.score((depth if name == "depth" else rgb)[None], batch=1)[0])
    # heatmap and box in the panel: depth map in 3D mode, else DINO. The DINO box centre lies on the defect in 56.0 %
    # of 632 defective images outside vlm_eval's sample, WRN50 38.3 %, CLIP 47.5 %, the three averaged 49.4 % (11-10)
    amap = maps[-1] if depth is not None else maps[1]
    clf = XGBClassifier()
    clf.load_model(CACHE / f"{'head3d' if depth is not None else 'head'}_{cls}.json")
    obj = (seg != bg_cluster(cls)).astype(np.int32)
    x = np.concatenate([segment_stats(m, obj, 2) for m in maps])
    prob = float(clf.predict_proba(x[None])[0, 1])
    torch.cuda.empty_cache()
    res = {"class": cls, "mode": "3D" if depth is not None else "camera",
           "defect_probability": round(prob, 3), "defective": prob >= threshold}
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
    if out:
        _panel(rgb, seg, amap, box, res, out)
    return res


def explain(path: Path, threshold: float | None = None, use_depth: bool = False) -> dict:
    cls = path.parts[-5]
    if threshold is None:
        threshold = demo_threshold(cls, "3D" if use_depth else "camera")
    if uid(path) in json.loads((CACHE / f"head_{cls}_train.json").read_text()):
        print(f"warning: {uid(path)} was used to train the demo head; pick another image")
    res = inspect(load_rgb(path), cls, load_depth(path) if use_depth else None, threshold,
                  RESULTS / f"explain_{cls}_{uid(path)}{'_3d' if use_depth else ''}.png")
    return {"image": str(path), **res}


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
    fig.suptitle(f"[{res['mode']} mode] {verdict}  (p = {res['defect_probability']:.2f})   {res.get('description', '')}", wrap=True)
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    res["panel"] = str(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "eval", "calibrate", "explain"])
    ap.add_argument("image", nargs="?", type=Path)
    ap.add_argument("--classes", nargs="+", default=CLASSES)
    ap.add_argument("--3d", dest="use_depth", action="store_true", help="explain: use the scanner depth (3D mode)")
    a = ap.parse_args()
    if not CLASSES:
        ap.error(f"no dataset in {MV}: download MVTec 3D-AD first (README, Run)")
    if unknown := set(a.classes) - set(CLASSES):
        ap.error(f"unknown classes {sorted(unknown)}; choose from {CLASSES}")
    if a.stage == "fit":
        for c in a.classes:
            fit(c)
    elif a.stage == "eval":
        report(a.classes)
    elif a.stage == "calibrate":
        t = json.loads(THRESHOLDS.read_text()) if THRESHOLDS.exists() else {}
        for c in a.classes:
            t[c] = calibrate(c)
            print(c, t[c], flush=True)
        THRESHOLDS.write_text(json.dumps(t, indent=1))
    else:
        img = a.image and a.image.resolve()   # explain reads the class and the depth scan from the dataset layout
        if not (img and img.is_file() and list(img.parents)[4:5] == [MV]):
            ap.error("explain needs a dataset image, e.g. data/mvtec3d/dowel/test/bent/rgb/000.png")
        print(json.dumps(explain(img, use_depth=a.use_depth), indent=2))


if __name__ == "__main__":
    assert group_of("hole") == "geometric" and group_of("combined") == "mixed"
    assert uid(Path("x/bagel/test/crack/rgb/000.png")) == "test_crack_000"
    b = np.zeros((SIZE, SIZE)); b[100:110, 50:60] = 1
    x0, y0, x1, y1 = defect_box(b)
    assert x0 <= 50 and y0 <= 100 and x1 >= 60 and y1 >= 110, (x0, y0, x1, y1)
    assert pick_threshold(np.array([0, 0, 1, 1]), np.array([.1, .12, .33, .9])) == 0.3   # 0.15-0.30 all perfect: nearest 0.5
    main()
