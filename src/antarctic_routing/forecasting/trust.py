"""Stage 8.4 - trust horizon: how long the forecast keeps *supported* skill.

    Delta(h) = E_baseline(h) - E_model(h)        (> 0: the model is better)

Uncertainty comes from a season-blocked bootstrap: whole test seasons are
resampled with replacement, so day-to-day autocorrelation inside a season does
not inflate confidence. The trust horizon is the last lead h for which *every*
lead 1..h has a lower confidence bound above ``min_delta``. It measures useful
predictive skill, not how far the model can produce output.
"""

from __future__ import annotations

import numpy as np


def trust_horizon(
    by_season: dict,
    model: str,
    baseline: str,
    n_boot: int = 2000,
    alpha: float = 0.05,
    min_delta: float = 0.0,
    seed: int = 0,
) -> dict:
    seasons = sorted(by_season[model], key=lambda s: int(s))
    m = np.array([by_season[model][s] for s in seasons], float)
    b = np.array([by_season[baseline][s] for s in seasons], float)
    d = b - m  # (S, H)
    n_seasons, n_leads = d.shape
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_seasons, size=(n_boot, n_seasons))
    boot = d[picks].mean(axis=1)  # (n_boot, H)
    lo = np.percentile(boot, 100 * alpha, axis=0)          # one-sided lower bound
    hi = np.percentile(boot, 100 * (1 - alpha), axis=0)
    horizon = 0
    for h in range(n_leads):
        if lo[h] > min_delta:
            horizon = h + 1
        else:
            break
    warnings = []
    if n_seasons < 5:
        warnings.append(f"only {n_seasons} test seasons: bootstrap intervals are coarse")
    return {
        "model": model,
        "baseline": baseline,
        "leads": list(range(1, n_leads + 1)),
        "delta": d.mean(axis=0).tolist(),
        "ci_low": lo.tolist(),
        "ci_high": hi.tolist(),
        "trust_horizon_days": horizon,
        "limited_by_max_lead": horizon == n_leads,
        "n_seasons": n_seasons,
        "n_boot": n_boot,
        "alpha": alpha,
        "min_delta": min_delta,
        "method": "season-blocked bootstrap, one-sided lower bound",
        "warnings": warnings,
    }
