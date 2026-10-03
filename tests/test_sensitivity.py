"""Fuel / speed assumption sensitivity (Stage 10.5)."""

import numpy as np

from _worlds import NX, NY, A, B, _vessel, _world
from antarctic_routing.validation.sensitivity import run_sensitivity


def _slush_world():
    conc = np.zeros((100, 3, NY, NX))
    conc[:, :, 2:9, 7:14] = 0.14   # below the vessel limit (no breach) but costly in fuel
    return _world(k=100, conc=conc)


def test_fuel_penalty_changes_the_recommended_route():
    res = run_sensitivity(_slush_world(), _vessel(lam=4.0, ice_k=0.7), A, B, lambdas=[0.0, 4.0, 200.0],
                          ks=[0.7], risk_budget=0.05, risk_weights=[0])
    cells = {row["lambda"]: row for row in res["grid"]}
    assert cells[0.0]["route_changed"] is False or cells[0.0]["deviation_km"] < 1.0
    assert cells[200.0]["route_changed"] is True and cells[200.0]["deviation_km"] > 10
    assert res["baseline"]["lambda"] == 4.0
    assert 0 < res["stable_fraction"] < 1


def test_open_water_recommendation_is_insensitive():
    res = run_sensitivity(_world(k=100), _vessel(), A, B, lambdas=[0.0, 4.0, 50.0], ks=[0.3, 0.7, 0.9],
                          risk_budget=0.05, risk_weights=[0])
    assert res["stable_fraction"] == 1.0
    assert len(res["grid"]) == 9
    assert all(not row["route_changed"] for row in res["grid"])


def test_speed_factor_changes_time_but_is_reported_per_setting():
    res = run_sensitivity(_slush_world(), _vessel(), A, B, lambdas=[4.0], ks=[0.0, 0.9],
                          risk_budget=0.05, risk_weights=[0])
    hours = {row["k"]: row["expected_hours"] for row in res["grid"]}
    assert hours[0.9] >= hours[0.0]


def test_window_mode_reports_departure_changes():
    conc = np.zeros((100, 6, NY, NX))
    conc[:, :2, :, 8:13] = 0.9
    world = _world(k=100, t=6, conc=conc)
    world.layer_source = ["observed", "forecast", "forecast", "climatology", "climatology", "climatology"]
    res = run_sensitivity(world, _vessel(), A, B, lambdas=[0.0, 4.0], ks=[0.7], risk_budget=0.05,
                          risk_weights=[0], offsets_days=[0, 1, 2, 3])
    assert all(row["departure_lead_days"] == 2 for row in res["grid"])
    assert res["stable_departure_fraction"] == 1.0
