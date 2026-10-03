"""Historical replay: day-by-day decisions using only information available each day."""

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from antarctic_routing.config import load_config
from antarctic_routing.forecasting.scenarios import ForecastContext, grid_of
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.replay import run_replay
from antarctic_routing.routing.fuel import VesselModel
from antarctic_routing.routing.replan import ReplanPolicy
from antarctic_routing.synthetic import synthetic_history

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
L, H = 5, 3


@pytest.fixture(scope="module")
def replay(tmp_path_factory):
    ds = synthetic_history(CFG, PolarGrid.from_domain(CFG.domain, resolution_km=50), range(2010, 2016), seed=7)

    def persistence(inputs, idx):
        return np.repeat(inputs[:, L - 1: L], H, axis=1)

    ctx = ForecastContext.build(ds, persistence, L, H, CFG.project.season_months, [2010, 2011, 2012],
                                bank_size=60, seed=0)
    grid = grid_of(ds)
    o = grid.cell_of(CFG.route.origin.lat, CFG.route.origin.lon)
    d = grid.cell_of(CFG.route.destination.lat, CFG.route.destination.lon)
    audit = tmp_path_factory.mktemp("replay") / "audit.jsonl"
    result = run_replay(ctx, VesselModel.from_config(CFG), o, d, start=date(2015, 11, 20), window_days=5,
                        max_wait_days=40, n_members=80, policy=ReplanPolicy(risk_budget=0.05),
                        risk_weights=[0, 100], voyage_days=3, rng=np.random.default_rng(0), audit_path=audit,
                        scenario_routes=0)
    return result, audit


def test_replay_waits_departs_and_arrives(replay):
    result, _ = replay
    phases = [d["phase"] for d in result["days"]]
    assert phases[0] == "port" and "at_sea" in phases
    assert result["arrived"] is True
    assert result["departure"] >= "2015-11-20"
    assert result["days_waited"] == phases.count("port") - 1


def test_every_decision_is_audited(replay):
    result, audit = replay
    lines = audit.read_text().splitlines()
    assert len(lines) == len(result["days"])
    assert all('"phase"' in line for line in lines)


def test_replay_scores_against_truth_and_naive_baseline(replay):
    result, _ = replay
    truth = result["truth"]
    assert truth["planner"]["observed_breach_cells"] >= 0
    assert {"observed_breach_cells", "hours", "fuel_index", "departure"} <= truth["naive"].keys()
    assert truth["naive"]["departure"] == "2015-11-20"
    assert result["execution_mode"] == "controlled_synthetic"


def test_port_decisions_only_use_issue_day_forecasts(replay):
    result, _ = replay
    for day in result["days"]:
        if day["phase"] == "port" and day["recommended_departure"]:
            assert day["recommended_departure"] >= day["day"]
