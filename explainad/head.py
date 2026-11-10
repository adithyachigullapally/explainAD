"""The SegAD head: a Boosted Random Forest (XGBoost) over per-segment statistics."""
from __future__ import annotations

import numpy as np

# Same hyper-parameters for every arm, seed and ablation, so only the input features differ.
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


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.5).astype(int)
    X = rng.normal(size=(400, 4))
    X[:, 2] += 2.0 * y   # only column 2 carries the label
    tr, te = slice(0, 300), slice(300, None)
    auc = 100 * roc_auc_score(y[te], fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
    assert auc > 80, auc
    print(f"head ok  auroc={auc:.1f}")
