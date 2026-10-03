import numpy as np
import pytest

from antarctic_routing.routing.fuel import (
    VesselModel,
    edge_travel_hours,
    fuel_index,
    ice_penalty,
    speed_in_ice_kmh,
)
from antarctic_routing.routing.hazard import combine_independent, exceedance_probability


def test_exceedance_probability_is_inclusive_over_members():
    members = np.array([[[0.1]], [[0.15]], [[0.5]], [[0.0]]])
    assert exceedance_probability(members, 0.15)[0, 0] == pytest.approx(0.5)


def test_exceedance_treats_nan_land_as_no_ice_probability():
    members = np.array([[np.nan, 0.9], [np.nan, 0.1]])
    p = exceedance_probability(members, 0.5)
    assert np.isnan(p[0]) and p[1] == pytest.approx(0.5)


def test_combine_independent_hand_value():
    assert combine_independent(0.1, 0.2) == pytest.approx(1 - 0.9 * 0.8)


def test_combine_rejects_invalid_probabilities():
    with pytest.raises(ValueError):
        combine_independent(np.array([1.2]), np.array([0.0]))


def test_quadratic_fuel_index_hand_value():
    assert fuel_index(10.0, 0.5, lam=4.0) == pytest.approx(10.0 * (1 + 4 * 0.25))


def test_open_water_fuel_index_equals_distance():
    assert fuel_index(37.5, 0.0, lam=4.0) == pytest.approx(37.5)


def test_piecewise_penalty_steepens_above_vessel_limit():
    tau = 0.15
    below = ice_penalty(tau, "piecewise", tau) - ice_penalty(tau - 0.05, "piecewise", tau)
    above = ice_penalty(tau + 0.05, "piecewise", tau) - ice_penalty(tau, "piecewise", tau)
    assert above > 3 * below


def test_speed_in_ice_reduction_and_floor():
    assert speed_in_ice_kmh(20.0, 0.5, k=0.7, v_min=5.0) == pytest.approx(20.0 * 0.65)
    assert speed_in_ice_kmh(20.0, 1.0, k=0.9, v_min=5.0) == pytest.approx(5.0)


def test_edge_travel_hours_with_following_and_opposing_current():
    assert edge_travel_hours(30.0, 20.0, 10.0) == pytest.approx(1.0)
    assert edge_travel_hours(30.0, 20.0, -10.0) == pytest.approx(3.0)


def test_edge_travel_is_impassable_when_current_overwhelms_vessel():
    assert edge_travel_hours(10.0, 5.0, -5.0) == float("inf")
    assert edge_travel_hours(10.0, 5.0, -9.0) == float("inf")


def test_vessel_model_from_config():
    from pathlib import Path

    from antarctic_routing.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    vm = VesselModel.from_config(cfg)
    assert vm.cruise_kmh == pytest.approx(12 * 1.852)
    assert vm.tau == cfg.vessel.max_ice_concentration
