from datetime import date, timedelta

import numpy as np
import pytest

from antarctic_routing.forecasting.baselines import (
    climatology_forecast,
    damped_anomaly_persistence,
    fit_anomaly_decay,
    persistence,
    threshold_probability,
)
from antarctic_routing.preprocessing.climatology import Climatology
from antarctic_routing.validation.metrics import (
    brier_score,
    masked_mae,
    masked_rmse,
    reliability_bins,
)

SEASON = [11, 12, 1, 2]


def _clim(value=0.5):
    dates = [date(2010, 11, 1) + timedelta(days=k) for k in range(120)]
    conc = np.full((120, 2, 2), value)
    return Climatology.fit(conc, dates, [2010], SEASON, window_days=0)


def test_persistence_returns_copy_of_latest_state():
    c = np.array([[0.2, 0.9]])
    out = persistence(c)
    assert np.array_equal(out, c) and out is not c


def test_climatology_forecast_uses_target_day():
    assert climatology_forecast(_clim(0.4), date(2026, 12, 1))[0, 0] == pytest.approx(0.4)


def test_damped_anomaly_persistence_hand_value():
    # mu = 0.5, C_t = 0.9 -> A = 0.4; rho = 0.5, h = 2 -> 0.5 + 0.25*0.4 = 0.6
    out = damped_anomaly_persistence(np.full((2, 2), 0.9), date(2026, 11, 10), 2, _clim(0.5), rho=0.5)
    assert out[0, 0] == pytest.approx(0.6)


def test_damped_anomaly_persistence_is_clipped():
    out = damped_anomaly_persistence(np.full((2, 2), 1.0), date(2026, 11, 10), 1, _clim(0.95), rho=1.0)
    assert out.max() <= 1.0


def test_damped_persistence_limits():
    clim = _clim(0.5)
    c = np.full((2, 2), 0.9)
    assert np.allclose(damped_anomaly_persistence(c, date(2026, 11, 10), 3, clim, rho=1.0), c)
    assert np.allclose(damped_anomaly_persistence(c, date(2026, 11, 10), 3, clim, rho=0.0), 0.5)


def test_fit_anomaly_decay_recovers_ar1_coefficient():
    rng = np.random.default_rng(0)
    rho_true, n = 0.8, 4000
    a = np.zeros((n, 3, 3))
    for t in range(1, n):
        a[t] = rho_true * a[t - 1] + rng.normal(0, 0.05, (3, 3))
    assert fit_anomaly_decay(a) == pytest.approx(rho_true, abs=0.03)


def test_threshold_probability_counts_members():
    members = np.array([[0.1], [0.2], [0.3], [0.4]])
    assert threshold_probability(members, 0.25)[0] == pytest.approx(0.5)
    assert threshold_probability(members, 0.2)[0] == pytest.approx(0.75)  # >= is inclusive


def test_masked_mae_and_rmse_ignore_excluded_cells():
    pred = np.array([0.0, 0.5, 1.0])
    obs = np.array([0.0, 0.0, 0.0])
    mask = np.array([True, True, False])
    assert masked_mae(pred, obs, mask) == pytest.approx(0.25)
    assert masked_rmse(pred, obs, mask) == pytest.approx(np.sqrt(0.125))


def test_masked_metrics_ignore_nan_observations():
    assert masked_mae(np.array([0.2, 0.4]), np.array([0.0, np.nan])) == pytest.approx(0.2)


def test_brier_score_hand_value():
    assert brier_score(np.array([0.9, 0.2]), np.array([1, 0])) == pytest.approx((0.01 + 0.04) / 2)


def test_reliability_bins_perfectly_calibrated():
    p = np.r_[np.full(100, 0.2), np.full(100, 0.8)]
    o = np.r_[np.r_[np.ones(20), np.zeros(80)], np.r_[np.ones(80), np.zeros(20)]]
    table = reliability_bins(p, o, n_bins=5)
    filled = [row for row in table if row["count"] > 0]
    assert len(filled) == 2
    for row in filled:
        assert row["mean_predicted"] == pytest.approx(row["observed_frequency"])
