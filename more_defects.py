"""Does the SegAD head (and its threshold) get better with more defective training images? Pre-registered
2026-10-24 before any number. Main results (10 defective images, SegAD protocol) are not changed by this.

Per class and seed (the 10 official seeds): 30 % of the defective test images are held out, and every setting is
tested on exactly those plus all test/good images. From the remaining 70 % the head is trained on validation/good
plus the first 10 (current protocol), 60 % or 70 % of the class's defective images (nested, one shuffle per seed).
Threshold: 5-fold out-of-fold probabilities on the training images, then pipeline.pick_threshold (best balanced
accuracy on the 0.05 grid, ties nearest 0.5); same rule in every setting. Arms: camera mode and 3D mode.
Reported: AUROC, balanced accuracy at 0.5 and at the fitted threshold, recall and false alarms at that threshold.
Run: python more_defects.py -> results/more_defects.txt, results/per_seed_more_defects.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import pipeline as pl
from explainad import head

ARMS = {"camera": "SAM object/bg + DINO + CLIP", "3D": "with 3D scanner: + real depth"}
HOLD_OUT = 0.3


def n_train(setting: str, n_bad: int) -> int:
    return 10 if setting == "10 images" else round(int(setting[:2]) / 100 * n_bad)


def bal_acc(y: np.ndarray, p: np.ndarray, t: float) -> float:
    return 100 * ((p[y == 1] >= t).mean() + (p[y == 0] < t).mean()) / 2


def oof_threshold(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    p = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        p[b] = head.fit(X[a], y[a], seed=seed).predict_proba(X[b])[:, 1]
    return pl.pick_threshold(y, p)


def run(cls: str) -> list[tuple]:
    todo = pl.paths(cls, "validation") + pl.paths(cls, "test")
    split = np.array([p.parts[-4] for p in todo])
    y = np.array([p.parts[-3] != "good" for p in todo]).astype(int)
    bad = np.flatnonzero((split == "test") & (y == 1))
    X, rows = pl.features(cls), []
    for seed in pl.SEEDS:
        order = np.random.default_rng(seed).permutation(bad)
        n_test = round(HOLD_OUT * len(bad))
        te = np.zeros(len(y), bool)
        te[order[:n_test]] = True
        te[(split == "test") & (y == 0)] = True
        pool = order[n_test:]
        for setting in ("10 images", "60 %", "70 %"):
            tr = split == "validation"
            tr[pool[:n_train(setting, len(bad))]] = True
            for mode, arm in ARMS.items():
                Xa = X[arm]
                p = head.fit(Xa[tr], y[tr], seed=seed).predict_proba(Xa[te])[:, 1]
                t = oof_threshold(Xa[tr], y[tr], seed)
                yt = y[te]
                rows.append((cls, seed, setting, mode, int(y[tr].sum()), 100 * roc_auc_score(yt, p),
                             bal_acc(yt, p, .5), t, bal_acc(yt, p, t),
                             100 * (p[yt == 1] >= t).mean(), 100 * (p[yt == 0] >= t).mean()))
    print(cls, flush=True)
    return rows


if __name__ == "__main__":
    assert n_train("10 images", 90) == 10 and n_train("60 %", 90) == 54 and n_train("70 %", 90) == 63
    assert bal_acc(np.array([0, 1]), np.array([.2, .8]), .5) == 100
    r = pd.DataFrame([row for c in pl.CLASSES for row in run(c)],
                     columns=["cls", "seed", "setting", "mode", "n_bad_train", "auroc", "bal_acc_05",
                              "threshold", "bal_acc_fitted", "recall", "false_alarm"])
    r.to_csv(pl.RESULTS / "per_seed_more_defects.csv", index=False)
    cols = ["auroc", "bal_acc_05", "bal_acc_fitted", "recall", "false_alarm"]
    per_seed = r.groupby(["mode", "setting", "seed"])[cols].mean()
    table = per_seed.groupby(["mode", "setting"]).agg(["mean", "std"]).round(1)
    per_cls = r.groupby(["mode", "cls", "setting"])[["auroc", "bal_acc_fitted"]].mean().unstack("setting").round(1)
    txt = "\n".join([__doc__.split("\n")[0], "Test = same held-out 30 % of defects + all test/good, per seed. "
                     "Mean (std over 10 seeds) of the class mean, %:", table.to_string(), "",
                     "Per class (mean over seeds):", per_cls.to_string()])
    (pl.RESULTS / "more_defects.txt").write_text(txt, encoding="utf-8")
    print(txt)
