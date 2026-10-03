"""Season-aware climatology, anomalies and chronological splits.

The austral summer season crosses New Year (e.g. Nov-Feb), so all of this works
in *seasons* (labelled by the year in which the season starts) and *day of
season*, never calendar years or day of year.

Leakage rule: climatology, normalisation statistics and anomaly decay are fitted
from training seasons only.

    mu_i(d)    = mean over training seasons of C_i(d)
    A_i,y(d)   = C_i,y(d) - mu_i(d)
    Z_i,y(d)   = A_i,y(d) / (sigma_i(d) + eps)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np


def _first_month(season_months: Sequence[int]) -> int:
    return season_months[0]


def season_of(d: date, season_months: Sequence[int]) -> int | None:
    """Season label (year in which the season starts) or None if out of season."""
    if d.month not in season_months:
        return None
    first = _first_month(season_months)
    return d.year if d.month >= first else d.year - 1


def day_of_season(d: date, season_months: Sequence[int]) -> int:
    season = season_of(d, season_months)
    if season is None:
        raise ValueError(f"{d} is outside the season {list(season_months)}")
    return (d - date(season, _first_month(season_months), 1)).days


def chronological_split(seasons: Sequence[int], n_val: int, n_test: int) -> dict[str, list[int]]:
    """Earliest seasons train, the next validate, the most recent test."""
    ordered = sorted(set(seasons))
    if len(ordered) - n_val - n_test < 1:
        raise ValueError("not enough seasons for a non-empty training split")
    n_train = len(ordered) - n_val - n_test
    return {
        "train": ordered[:n_train],
        "val": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val :],
    }


@dataclass(frozen=True)
class Climatology:
    """Per-cell mean/std indexed by day of season."""

    mean: np.ndarray  # (n_days, ny, nx)
    std: np.ndarray
    count: np.ndarray
    season_months: tuple[int, ...]
    train_seasons: tuple[int, ...]
    eps: float = 1e-3

    @classmethod
    def fit(
        cls,
        conc: np.ndarray,
        dates: Sequence[date],
        train_seasons: Sequence[int],
        season_months: Sequence[int],
        window_days: int = 7,
    ) -> Climatology:
        conc = np.asarray(conc, float)
        if conc.shape[0] != len(dates):
            raise ValueError("conc and dates length mismatch")
        train = set(train_seasons)
        n_days = 366
        sums = np.zeros((n_days, *conc.shape[1:]))
        sq = np.zeros_like(sums)
        cnt = np.zeros_like(sums)
        for field, d in zip(conc, dates):
            s = season_of(d, season_months)
            if s is None or s not in train:
                continue
            k = day_of_season(d, season_months)
            valid = np.isfinite(field)
            for off in range(-window_days, window_days + 1):
                j = k + off
                if 0 <= j < n_days:
                    sums[j][valid] += field[valid]
                    sq[j][valid] += field[valid] ** 2
                    cnt[j][valid] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(cnt > 0, sums / cnt, np.nan)
            var = np.where(cnt > 0, sq / cnt - mean**2, np.nan)
        std = np.sqrt(np.clip(var, 0, None))
        return cls(mean, std, cnt, tuple(season_months), tuple(sorted(train)))

    def _index(self, d: date) -> int:
        return day_of_season(d, self.season_months)

    def mean_for(self, d: date) -> np.ndarray:
        return self.mean[self._index(d)]

    def std_for(self, d: date) -> np.ndarray:
        return self.std[self._index(d)]

    def anomaly(self, conc: np.ndarray, d: date) -> np.ndarray:
        return np.asarray(conc, float) - self.mean_for(d)

    def standardized(self, conc: np.ndarray, d: date) -> np.ndarray:
        return self.anomaly(conc, d) / (self.std_for(d) + self.eps)
