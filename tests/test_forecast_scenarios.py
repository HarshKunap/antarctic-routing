"""Forecast-driven joint scenarios (Stage 8.1-8.2)."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from antarctic_routing.config import load_config
from antarctic_routing.forecasting.scenarios import ForecastContext
from antarctic_routing.preprocessing.climatology import day_of_season
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.synthetic import synthetic_history

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
SEASON = CFG.project.season_months
L, H = 5, 3
TRAIN = [2010, 2011, 2012]
ISSUE = date(2014, 12, 10)


@pytest.fixture(scope="module")
def history():
    grid = PolarGrid.from_domain(CFG.domain, resolution_km=50)
    return synthetic_history(CFG, grid, seasons=range(2010, 2016), seed=5)


def _persistence(ds):
    conc = np.nan_to_num(ds["ice_concentration"].values, nan=0.0)

    def predict(inputs, idx):
        return np.repeat(inputs[:, L - 1: L], H, axis=1)  # uses only the input window

    return predict, conc


def _context(ds, seed=0):
    predictor, _ = _persistence(ds)
    return ForecastContext.build(ds, predictor, L, H, SEASON, TRAIN, bank_size=50, seed=seed)


def test_layers_are_observed_then_forecast_then_climatology(history):
    ctx = _context(history)
    world = ctx.scenarios(ISSUE, n_days=7, n_members=12, rng=np.random.default_rng(0))
    assert world.conc.shape == (12, 7, *history["land_mask"].shape)
    assert world.layer_source == ["observed", "forecast", "forecast", "forecast",
                                  "climatology", "climatology", "climatology"]
    t = ctx.index_of(ISSUE)
    obs = history["ice_concentration"].values[t]
    ocean = ~history["land_mask"].values
    assert np.allclose(world.conc[:, 0][:, ocean], obs[ocean][None])
    assert np.isnan(world.conc[:, :, ~ocean]).all()
    vals = world.conc[:, :, ocean]
    assert vals.min() >= 0 and vals.max() <= 1
    assert world.start.date() == ISSUE
    assert world.execution_mode == "controlled_synthetic"
    assert world.current_x.shape == (7, *ocean.shape) and world.wind_x is not None


def test_no_future_information_is_used(history):
    """Overwriting every observation after the issue date must not change the scenarios."""
    ctx_a = _context(history)
    a = ctx_a.scenarios(ISSUE, 7, 12, np.random.default_rng(1))
    tampered = history.copy(deep=True)
    future = tampered["time"].values > np.datetime64(ISSUE)
    tampered["ice_concentration"].values[future] = 1.0
    ctx_b = _context(tampered)
    b = ctx_b.scenarios(ISSUE, 7, 12, np.random.default_rng(1))
    assert np.array_equal(a.conc, b.conc, equal_nan=True)


def test_beyond_horizon_members_are_coherent_historical_anomaly_sequences(history):
    ctx = _context(history)
    world = ctx.scenarios(ISSUE, 8, 10, np.random.default_rng(2))
    ocean = ~history["land_mask"].values
    for k in range(10):
        season = world.meta["climatology_seasons"][k]
        assert season in TRAIN
        for h in range(H + 1, 8):
            day = ISSUE + timedelta(days=h)
            expected = np.clip(ctx.clim.mean_for(day) + ctx.anomaly(season, day), 0, 1)
            assert np.allclose(world.conc[k, h][ocean], expected[ocean], atol=1e-6)


def test_issue_without_enough_history_is_rejected(history):
    ctx = _context(history)
    with pytest.raises(ValueError, match="history"):
        ctx.scenarios(date(2014, 11, 2), 5, 4, np.random.default_rng(0))


def test_bank_and_climatology_come_from_training_seasons_only(history):
    ctx = _context(history)
    assert ctx.train_seasons == TRAIN
    assert set(ctx.bank_seasons) <= set(TRAIN)
    assert day_of_season(ISSUE, SEASON) >= 0
