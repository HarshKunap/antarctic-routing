"""Stage 6 - hazard layers.

    p_ice(i,t)  = P(C(i,t) >= tau_v)                (ensemble frequency)
    p_haz(i,t)  = 1 - (1 - p_ice)(1 - q_berg)        (independence approximation)

The independence formula is used only to *guide* the route search. Final route
risk is always evaluated jointly, scenario by scenario (see ``evaluate.py``),
because neighbouring cells and the two hazards are strongly correlated.
"""

from __future__ import annotations

import numpy as np


def exceedance_probability(members: np.ndarray, tau: float) -> np.ndarray:
    """Fraction of valid ensemble members (axis 0) with concentration >= tau.

    Cells with no valid members (e.g. land, all NaN) return NaN.
    """
    m = np.asarray(members, float)
    valid = np.isfinite(m)
    hits = np.sum(valid & (np.nan_to_num(m, nan=-1.0) >= tau), axis=0)
    n = np.sum(valid, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, hits / np.maximum(n, 1), np.nan)


def combine_independent(p_ice, q_berg):
    """1 - (1 - p_ice)(1 - q_berg); valid only if the hazards are independent."""
    p = np.asarray(p_ice, float)
    q = np.asarray(q_berg, float)
    for name, arr in (("p_ice", p), ("q_berg", q)):
        finite = arr[np.isfinite(arr)]
        if finite.size and (finite.min() < 0 or finite.max() > 1):
            raise ValueError(f"{name} must be within [0, 1]")
    out = 1.0 - (1.0 - p) * (1.0 - q)
    return float(out) if out.ndim == 0 else out
