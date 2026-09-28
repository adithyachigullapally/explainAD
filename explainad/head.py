"""The SegAD head: a Boosted Random Forest over per-segment statistics, plus metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# Same hyper-parameters for every run and every ablation. The whole experiment is
# "does adding depth columns help", so anything else must stay frozen.
PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="auc",
    tree_method="hist",
)


def fit(X: np.ndarray, y: np.ndarray, seed: int = 0):
    from xgboost import XGBClassifier

    pos = max(int((y == 1).sum()), 1)
    clf = XGBClassifier(
        **PARAMS, random_state=seed, scale_pos_weight=float((y == 0).sum()) / pos
    )
    clf.fit(X, y)
    return clf


def fpr_at_tpr(y: np.ndarray, score: np.ndarray, target_tpr: float = 0.95) -> float:
    """Fraction of good parts wrongly rejected when 95% of bad parts are caught."""
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.interp(target_tpr, tpr, fpr))


def evaluate(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "auroc": 100 * roc_auc_score(y, score),
        "fpr@95tpr": 100 * fpr_at_tpr(y, score),
    }


def select(X: np.ndarray, names: list[str], streams: tuple[str, ...]) -> np.ndarray:
    """Keep only the columns belonging to the given feature streams.

    This is the ablation: GeoAD minus its depth columns is exactly SegAD, same rows,
    same seed, same hyper-parameters.
    """
    keep = [i for i, n in enumerate(names) if n.split("_")[0] in streams]
    assert keep, f"no columns for {streams} in {sorted({n.split('_')[0] for n in names})}"
    return X[:, keep]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < 0.5).astype(int)
    # rgb columns are pure noise; depth columns actually carry the label
    names = ["rgb_seg0_mean", "rgb_seg1_mean", "depth_seg0_mean", "depth_seg1_mean"]
    X = rng.normal(size=(n, 4))
    X[:, 2] += 2.0 * y

    assert select(X, names, ("rgb",)).shape == (n, 2)
    assert select(X, names, ("rgb", "depth")).shape == (n, 4)

    tr, te = slice(0, 300), slice(300, None)
    m_rgb = evaluate(y[te], fit(select(X, names, ("rgb",))[tr], y[tr]).predict_proba(
        select(X, names, ("rgb",))[te])[:, 1])
    m_all = evaluate(y[te], fit(select(X, names, ("rgb", "depth"))[tr], y[tr]).predict_proba(
        select(X, names, ("rgb", "depth"))[te])[:, 1])
    assert m_all["auroc"] > m_rgb["auroc"] + 10, (m_rgb, m_all)
    assert 0 <= m_all["fpr@95tpr"] <= 100
    print(f"head ok  rgb={m_rgb['auroc']:.1f}  rgb+depth={m_all['auroc']:.1f}")
