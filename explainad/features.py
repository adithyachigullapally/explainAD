"""SegAD feature extraction: per-segment statistics of a pixel-wise map.

SegAD Sec. 4: for every map f_k and every segment s_l compute the 99.5% quantile,
skewness, kurtosis and mean over the pixels where s_l(x) = 1. The feature vector is
the concatenation, length K * L * 4.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

STATS = ("q995", "skew", "kurt", "mean")


def segment_stats(amap: np.ndarray, segments: np.ndarray, n_segments: int) -> np.ndarray:
    """(L*4,) feature vector for one map.

    amap: (H, W) float. segments: (H, W) int in [0, n_segments).
    Empty segments yield zeros rather than NaN, so a segmentation map with an unused
    label cannot poison the whole feature matrix.
    """
    assert amap.shape == segments.shape, (amap.shape, segments.shape)
    out = np.zeros((n_segments, len(STATS)), dtype=np.float64)
    flat_a, flat_s = amap.ravel(), segments.ravel()
    order = np.argsort(flat_s, kind="stable")
    flat_a, flat_s = flat_a[order], flat_s[order]
    bounds = np.searchsorted(flat_s, np.arange(n_segments + 1))
    for l in range(n_segments):
        v = flat_a[bounds[l]:bounds[l + 1]]
        if v.size < 4:  # skew/kurtosis are meaningless below a handful of pixels
            continue
        out[l] = (np.quantile(v, 0.995), stats.skew(v), stats.kurtosis(v), v.mean())
    return np.nan_to_num(out.ravel(), nan=0.0, posinf=0.0, neginf=0.0)


def feature_names(stream: str, n_segments: int) -> list[str]:
    return [f"{stream}_seg{l}_{s}" for l in range(n_segments) for s in STATS]


def build_matrix(maps: dict[str, np.ndarray], segments: np.ndarray, n_segments: int):
    """maps: {stream_name: (N, H, W)} -> (X, names), streams concatenated in key order."""
    cols, names = [], []
    for stream in sorted(maps):
        arr = maps[stream]
        cols.append(np.stack([segment_stats(a, segments, n_segments) for a in arr]))
        names += feature_names(stream, n_segments)
    return np.concatenate(cols, axis=1), names


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    seg = np.zeros((64, 64), dtype=int)
    seg[32:] = 1
    seg[:2, :2] = 2  # 4 px: small but not degenerate
    amap = rng.normal(size=(64, 64))
    amap[32:] += 5.0  # segment 1 is anomalous

    f = segment_stats(amap, seg, 3)
    assert f.shape == (12,), f.shape
    assert np.isfinite(f).all()
    # mean of segment 1 must be ~5 higher than segment 0
    m0, m1 = f[0 * 4 + 3], f[1 * 4 + 3]
    assert m1 - m0 > 4.0, (m0, m1)

    # an empty segment must give zeros, not NaN
    f2 = segment_stats(amap, np.zeros_like(seg), 3)
    assert np.allclose(f2[4:], 0.0)

    X, names = build_matrix({"rgb": amap[None], "depth": amap[None] * 2}, seg, 3)
    assert X.shape == (1, 24) and len(names) == 24
    assert names[0] == "depth_seg0_q995"  # sorted key order
    print("features ok")
