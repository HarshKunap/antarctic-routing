"""Multi-start replay backtest against naive and fixed ice-edge-buffer baselines (Stage 10.4)."""

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from antarctic_routing.config import load_config
from antarctic_routing.forecasting.scenarios import ForecastContext, grid_of
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.routing.fuel import VesselModel
from antarctic_routing.routing.replan import ReplanPolicy
from antarctic_routing.synthetic import synthetic_history
from antarctic_routing.validation.backtest import buffer_baseline, run_backtest

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
L, H = 5, 3
TRAIN = [2010, 2011, 2012]
STARTS = [date(2014, 11, 18), date(2015, 11, 18), date(2015, 12, 2)]


@pytest.fixture(scope="module")
def setup():
    ds = synthetic_history(CFG, PolarGrid.from_domain(CFG.domain, resolution_km=50), range(2010, 2016), seed=11)
    ctx = ForecastContext.build(ds, lambda x, i: np.repeat(x[:, L - 1: L], H, axis=1), L, H,
                                CFG.project.season_months, TRAIN, bank_size=60, seed=0)
    grid = grid_of(ds)
    o = grid.cell_of(CFG.route.origin.lat, CFG.route.origin.lon)
    d = grid.cell_of(CFG.route.destination.lat, CFG.route.destination.lon)
    return ctx, o, d


@pytest.fixture(scope="module")
def backtest(setup):
    ctx, o, d = setup
    return run_backtest(ctx, VesselModel.from_config(CFG), o, d, STARTS, window_days=4, max_wait_days=30,
                        n_members=80, policy=ReplanPolicy(risk_budget=0.05), risk_weights=[0, 100], voyage_days=3,
                        buffer_km=60.0, seed=0, scenario_routes=0)


def test_every_start_is_scored_for_every_method(backtest):
    assert len(backtest["voyages"]) == len(STARTS) * 3
    assert {v["method"] for v in backtest["voyages"]} == {"planner", "naive", "ice_edge_buffer"}
    for row in backtest["voyages"]:
        assert {"start", "season", "departure", "days_waited", "hazard_hours", "observed_breach_cells",
                "breached", "fuel_index", "sail_hours", "elapsed_hours"} <= row.keys()


def test_aggregate_metrics_per_method(backtest):
    agg = backtest["summary"]
    for method in ("planner", "naive", "ice_edge_buffer"):
        m = agg[method]
        assert m["n"] == len(STARTS)
        assert 0.0 <= m["breach_rate"] <= 1.0
        assert m["mean_days_waited"] >= 0
    assert agg["naive"]["mean_days_waited"] == 0


def test_predicted_vs_realised_route_risk_is_reported(backtest):
    rel = backtest["planner_risk_check"]
    assert rel["n_departures"] == len(STARTS)
    assert 0 <= rel["mean_predicted_p_breach"] <= 1 and 0 <= rel["realised_breach_rate"] <= 1


def test_buffer_baseline_keeps_its_distance_from_observed_ice_at_departure(setup):
    ctx, o, d = setup
    out = buffer_baseline(ctx, VesselModel.from_config(CFG), o, d, date(2015, 12, 2), buffer_km=60.0,
                          max_wait_days=30)
    grid = grid_of(ctx.ds)
    obs = ctx.ds["ice_concentration"].values[ctx.index_of(date.fromisoformat(out["departure"]))]
    ice = np.argwhere(np.nan_to_num(obs) >= CFG.vessel.max_ice_concentration)
    if ice.size:
        xy_ice = np.column_stack([grid.x[ice[:, 1]], grid.y[ice[:, 0]]])
        for r, c in out["cells"][1:-1]:
            nearest = np.min(np.hypot(xy_ice[:, 0] - grid.x[c], xy_ice[:, 1] - grid.y[r]))
            assert nearest >= 60_000 - grid.resolution_m


def test_backtest_refuses_training_seasons(setup):
    ctx, o, d = setup
    with pytest.raises(ValueError, match="training"):
        run_backtest(ctx, VesselModel.from_config(CFG), o, d, [date(2011, 11, 20)], 4, 30, 80,
                     ReplanPolicy(0.05), [0], 3)
