"""Hand-crafted window features for the classical baseline.

Nothing fancy - the usual time/frequency statistics per channel plus a
magnitude channel for every 3-axis sensor group and the pairwise axis
correlations. ~20 features per channel. This is roughly what people used
before deep models and it is still a strong baseline on this kind of data.
"""

from __future__ import annotations

import numpy as np

from .data import FS


def _add_magnitudes(X: np.ndarray) -> np.ndarray:
    # channels come in xyz triplets
    C = X.shape[1]
    mags = [np.linalg.norm(X[:, i:i + 3], axis=1, keepdims=True) for i in range(0, C - C % 3, 3)]
    return np.concatenate([X, *mags], axis=1)


def window_features(X: np.ndarray) -> np.ndarray:
    """X: (N, C, L) -> (N, F)"""
    X = _add_magnitudes(X.astype(np.float64))
    N, C, L = X.shape

    mean = X.mean(-1)
    std = X.std(-1)
    mn, mx = X.min(-1), X.max(-1)
    p10, p25, med, p75, p90 = np.percentile(X, [10, 25, 50, 75, 90], axis=-1)
    centered = X - mean[..., None]
    skew = (centered ** 3).mean(-1) / (std ** 3 + 1e-9)
    kurt = (centered ** 4).mean(-1) / (std ** 4 + 1e-9)
    energy = (X ** 2).mean(-1)
    mad = np.abs(centered).mean(-1)
    zero_cross = (np.diff(np.sign(centered), axis=-1) != 0).mean(-1)
    jerk = np.abs(np.diff(X, axis=-1)).mean(-1)

    # frequency domain
    spec = np.abs(np.fft.rfft(centered * np.hanning(L), axis=-1)) ** 2
    freqs = np.fft.rfftfreq(L, d=1.0 / FS)
    spec_sum = spec.sum(-1) + 1e-12
    dom_freq = freqs[spec[..., 1:].argmax(-1) + 1]
    centroid = (spec * freqs).sum(-1) / spec_sum
    p = spec / spec_sum[..., None]
    spec_entropy = -(p * np.log(p + 1e-12)).sum(-1)
    band = lambda lo, hi: spec[..., (freqs >= lo) & (freqs < hi)].sum(-1) / spec_sum  # noqa: E731
    b1, b2, b3 = band(0.3, 1.5), band(1.5, 3.5), band(3.5, 8.0)

    feats = [mean, std, mn, mx, p10, p25, med, p75, p90, skew, kurt, energy, mad,
             zero_cross, jerk, dom_freq, centroid, spec_entropy, b1, b2, b3]
    out = np.concatenate(feats, axis=1)

    # correlation between axes inside each raw sensor triplet
    raw_c = C - (C // 4)  # number of original channels (before magnitudes)
    corrs = []
    for g in range(0, raw_c - raw_c % 3, 3):
        z = centered[:, g:g + 3] / (std[:, g:g + 3, None] + 1e-9)
        for a, b in ((0, 1), (0, 2), (1, 2)):
            corrs.append((z[:, a] * z[:, b]).mean(-1, keepdims=True))
    if corrs:
        out = np.concatenate([out, *corrs], axis=1)
    return np.nan_to_num(out).astype(np.float32)
