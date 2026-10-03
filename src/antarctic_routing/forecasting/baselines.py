"""Stage 4 - required non-neural baselines.

Any learned forecast must beat these at every lead time before it is trusted.

    Persistence:              C_hat(t+h) = C(t)
    Climatology:              C_hat(t+h) = mu(d_{t+h})
    Damped anomaly persist.:  C_hat(t+h) = clip(mu(d_{t+h}) + rho^h (C(t) - mu(d_t)), 0, 1)
    Threshold probability:    p_hat = (1/K) sum_k 1[C^(k) >= tau]
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from antarctic_routing.preprocessing.climatology import Climatology


def persistence(conc_t: np.ndarray) -> np.ndarray:
    return np.array(conc_t, dtype=float, copy=True)


def climatology_forecast(clim: Climatology, target: date) -> np.ndarray:
    return np.array(clim.mean_for(target), copy=True)


def damped_anomaly_persistence(
    conc_t: np.ndarray, issue_date: date, lead_days: int, clim: Climatology, rho: float
) -> np.ndarray:
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be within [0, 1]")
    anomaly = clim.anomaly(conc_t, issue_date)
    target = issue_date + timedelta(days=lead_days)
    return np.clip(clim.mean_for(target) + rho**lead_days * anomaly, 0.0, 1.0)


def fit_anomaly_decay(anomalies: np.ndarray) -> float:
    """Least-squares lag-1 decay ``rho`` from a daily anomaly sequence (T, ...).

    rho = sum(A_t A_{t+1}) / sum(A_t^2), clipped to [0, 1]. Pass training-season
    anomalies only.
    """
    a = np.asarray(anomalies, float)
    x, y = a[:-1].ravel(), a[1:].ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    denom = float(np.sum(x[ok] ** 2))
    if denom == 0:
        raise ValueError("anomalies have zero variance")
    return float(np.clip(np.sum(x[ok] * y[ok]) / denom, 0.0, 1.0))


def threshold_probability(members: np.ndarray, tau: float) -> np.ndarray:
    """Ensemble exceedance frequency along axis 0 (inclusive threshold)."""
    return np.mean(np.asarray(members) >= tau, axis=0)
