from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from antarctic_routing.config import (
    ProjectConfig,
    load_config,
    min_scenarios_for_budget,
    wilson_upper_bound,
)

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def _raw() -> dict:
    return yaml.safe_load(REPO_CONFIG.read_text())


def test_repository_config_loads_and_validates():
    cfg = load_config(REPO_CONFIG)
    assert cfg.domain.crs == "EPSG:3031"
    assert 0 < cfg.routing.risk_budget < 1
    assert cfg.route.origin.lat < 0


def test_points_use_explicit_lat_lon_keys_not_ambiguous_lists():
    raw = _raw()
    raw["route"]["origin"] = [-56.0, -66.0]
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(raw)


def test_latitude_out_of_range_rejected():
    raw = _raw()
    raw["route"]["origin"]["lat"] = -95.0
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(raw)


def test_route_endpoints_must_lie_inside_domain():
    raw = _raw()
    raw["route"]["destination"] = {"lat": -40.0, "lon": -60.0}
    with pytest.raises(ValidationError, match="outside the domain"):
        ProjectConfig.model_validate(raw)


def test_departure_window_must_be_ordered():
    raw = _raw()
    raw["route"]["departure_start"] = "2027-03-01"
    raw["route"]["departure_end"] = "2026-11-01"
    with pytest.raises(ValidationError, match="departure_start"):
        ProjectConfig.model_validate(raw)


def test_season_months_must_be_valid_and_unique():
    raw = _raw()
    raw["project"]["season_months"] = [11, 11, 13]
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(raw)


def test_ice_concentration_limit_is_a_fraction():
    raw = _raw()
    raw["vessel"]["max_ice_concentration"] = 15  # percent, not fraction
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(raw)


def test_too_few_scenarios_can_never_certify_the_risk_budget():
    # With 20 scenarios and zero breaches the 95% Wilson upper bound is ~0.16,
    # so a 5% budget could never be demonstrated.
    raw = _raw()
    raw["forecast"]["route_scenarios"] = 20
    raw["routing"]["risk_budget"] = 0.05
    with pytest.raises(ValidationError, match="route_scenarios"):
        ProjectConfig.model_validate(raw)


def test_wilson_upper_bound_hand_calculated():
    # Zero events out of n: upper = z^2 / (n + z^2)
    z = 1.959963984540054
    assert wilson_upper_bound(0, 20) == pytest.approx(z**2 / (20 + z**2), rel=1e-9)
    # 5 of 100: known value ~0.1118
    assert wilson_upper_bound(5, 100) == pytest.approx(0.11176, abs=5e-5)


def test_min_scenarios_for_budget():
    n = min_scenarios_for_budget(0.05)
    assert wilson_upper_bound(0, n) <= 0.05
    assert wilson_upper_bound(0, n - 1) > 0.05
    assert n == 73


def test_cruise_speed_conversion():
    cfg = load_config(REPO_CONFIG)
    assert cfg.vessel.cruise_speed_kmh == pytest.approx(cfg.vessel.cruise_speed_knots * 1.852)
