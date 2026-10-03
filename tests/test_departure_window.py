"""Departure window planned from a single forecast issue (Stage 8.3-8.5)."""

import numpy as np

from _worlds import NX, NY, A, B, _vessel, _world
from antarctic_routing.routing.departure import plan_from_issue

SOURCES = ["observed", "forecast", "forecast", "climatology", "climatology", "climatology"]


def _window_world():
    conc = np.zeros((100, 6, NY, NX))
    conc[:, :2, :, 8:13] = 0.9          # ice band blocks the strait on days 0-1, clears from day 2
    world = _world(k=100, t=6, conc=conc)
    world.layer_source = list(SOURCES)
    return world


def test_options_use_the_layers_after_their_departure_offset():
    sweep = plan_from_issue(_window_world(), [0, 1, 2, 3], _vessel(), A, B, risk_budget=0.05,
                            risk_weights=[0, 100], n_scenario_routes=0)
    assert [o.lead_days for o in sweep.options] == [0, 1, 2, 3]
    assert [o.feasible for o in sweep.options] == [False, False, True, True]
    assert sweep.selected.lead_days == 2
    assert sweep.options[2].departure.isoformat() == "2026-12-03"


def test_options_report_forecast_support_and_trust():
    sweep = plan_from_issue(_window_world(), [1, 2, 3], _vessel(), A, B, risk_budget=0.05, risk_weights=[0, 100],
                            n_scenario_routes=0, trust_horizon_days=2)
    by_lead = {o.lead_days: o for o in sweep.options}
    assert by_lead[2].forecast_fraction == 1.0 and by_lead[2].support == "forecast-supported"
    assert by_lead[3].forecast_fraction == 0.0 and by_lead[3].support == "climatology-dominated"
    assert by_lead[2].within_trust_horizon is True
    assert by_lead[3].within_trust_horizon is False


def test_require_trusted_excludes_untrusted_dates_from_selection():
    sweep = plan_from_issue(_window_world(), [2, 3], _vessel(), A, B, risk_budget=0.05, risk_weights=[0, 100],
                            n_scenario_routes=0, trust_horizon_days=1, require_trusted=True)
    assert sweep.selected is None
    assert "trust horizon" in sweep.explanation


def test_window_must_fit_inside_the_scenario_horizon():
    import pytest

    with pytest.raises(ValueError, match="horizon"):
        plan_from_issue(_window_world(), [6], _vessel(), A, B, risk_budget=0.05, risk_weights=[0])
