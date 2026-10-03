"""Forecast verification metrics (Stage 10).

MAE / RMSE are masked so land, missing observations and excluded cells never
contribute. Brier score and reliability tables evaluate probabilities.
"""

from __future__ import annotations

import numpy as np


def _valid(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    valid = np.isfinite(pred) & np.isfinite(obs)
    if mask is not None:
        valid &= np.asarray(mask, bool)
    if not valid.any():
        raise ValueError("no valid cells to score")
    return valid


def masked_mae(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray | None = None) -> float:
    v = _valid(pred, obs, mask)
    return float(np.mean(np.abs(np.asarray(pred, float)[v] - np.asarray(obs, float)[v])))


def masked_rmse(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray | None = None) -> float:
    v = _valid(pred, obs, mask)
    return float(np.sqrt(np.mean((np.asarray(pred, float)[v] - np.asarray(obs, float)[v]) ** 2)))


def brier_score(prob: np.ndarray, outcome: np.ndarray) -> float:
    """BS = mean((p - o)^2), o in {0, 1}. Lower is better."""
    p, o = np.asarray(prob, float).ravel(), np.asarray(outcome, float).ravel()
    if p.shape != o.shape or p.size == 0:
        raise ValueError("prob and outcome must be non-empty and the same shape")
    return float(np.mean((p - o) ** 2))


def reliability_bins(prob: np.ndarray, outcome: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Mean predicted probability vs observed frequency per probability bin."""
    p, o = np.asarray(prob, float).ravel(), np.asarray(outcome, float).ravel()
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        rows.append(
            {
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "count": n,
                "mean_predicted": float(p[sel].mean()) if n else None,
                "observed_frequency": float(o[sel].mean()) if n else None,
            }
        )
    return rows
