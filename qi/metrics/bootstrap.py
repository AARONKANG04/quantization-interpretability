"""Confidence intervals that match the plan: paired bootstrap over items for accuracy deltas, cluster bootstrap
over documents for per-token KL, and the flips metric of Dutta et al. (2024)."""
from __future__ import annotations

import numpy as np


def _ci(samples: np.ndarray, ci: float):
    lo, hi = np.percentile(samples, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def paired_bootstrap_delta(correct_a, correct_b, n_boot: int = 10_000, seed: int = 0, ci: float = 0.95, batch: int = 500) -> dict:
    """Accuracy of b minus accuracy of a on the same items, with a paired bootstrap CI (resampling items)."""
    a = np.asarray(correct_a, dtype=np.float64)
    b = np.asarray(correct_b, dtype=np.float64)
    assert a.shape == b.shape, "paired bootstrap needs the same items in the same order"
    n = a.size
    d = b - a
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot)
    for s in range(0, n_boot, batch):
        idx = rng.integers(0, n, size=(min(batch, n_boot - s), n))
        samples[s : s + idx.shape[0]] = d[idx].mean(axis=1)
    lo, hi = _ci(samples, ci)
    return {"n": int(n), "acc_a": float(a.mean()), "acc_b": float(b.mean()), "delta": float(d.mean()), "lo": lo, "hi": hi, "ci": ci}


def cluster_bootstrap_mean(values, clusters, n_boot: int = 2000, seed: int = 0, ci: float = 0.95) -> dict:
    """Mean of values with a CI from resampling whole clusters (documents), not individual tokens."""
    v = np.asarray(values, dtype=np.float64)
    c = np.asarray(clusters)
    uniq, inv = np.unique(c, return_inverse=True)
    sums = np.bincount(inv, weights=v, minlength=uniq.size)
    counts = np.bincount(inv, minlength=uniq.size).astype(np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, uniq.size, size=uniq.size)
        samples[i] = sums[pick].sum() / counts[pick].sum()
    lo, hi = _ci(samples, ci)
    return {"n_values": int(v.size), "n_clusters": int(uniq.size), "mean": float(v.mean()), "lo": lo, "hi": hi, "ci": ci}


def flips(correct_a, correct_b) -> dict:
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    return {
        "correct_to_incorrect": float((a & ~b).mean()),
        "incorrect_to_correct": float((~a & b).mean()),
        "total": float((a != b).mean()),
        "n": int(a.size),
    }
