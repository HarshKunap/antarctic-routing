from datetime import date, timedelta

import numpy as np

from _worlds import NX, NY, A, B, _vessel, _world
from antarctic_routing.routing.departure import sweep_departures

DATES = [date(2026, 12, 1) + timedelta(days=i) for i in range(4)]


def _factory(kind_by_date):
    def make(d):
        conc = np.zeros((100, 3, NY, NX))
        kind = kind_by_date[d]
        if kind == "blocked":
            conc[:, :, :, 8:13] = 0.9          # ice band across the whole domain
        elif kind == "slushy":
            conc[:, :, :, :] = 0.1             # below the limit, but costs fuel
        return _world(k=100, conc=conc)
    return make


def _sweep(kinds):
    return sweep_departures(
        DATES, _factory(dict(zip(DATES, kinds, strict=True))), _vessel(), A, B,
        risk_budget=0.05, risk_weights=[0, 100], n_scenario_routes=0,
    )


def test_each_date_is_evaluated_and_flagged():
    sweep = _sweep(["blocked", "blocked", "slushy", "clear"])
    assert [o.departure for o in sweep.options] == DATES
    assert [o.feasible for o in sweep.options] == [False, False, True, True]
    assert sweep.options[0].p_breach == 1.0


def test_selection_rule_is_min_expected_fuel_among_feasible_dates():
    sweep = _sweep(["blocked", "clear", "slushy", "clear"])
    assert sweep.selected.departure == DATES[1]  # tie on fuel/time -> earliest date
    assert "min expected fuel" in sweep.rule


def test_lower_fuel_date_wins_over_earlier_date():
    sweep = _sweep(["slushy", "clear", "blocked", "blocked"])
    assert sweep.selected.departure == DATES[1]
    fuel = {o.departure: o.expected_fuel for o in sweep.options}
    assert fuel[DATES[1]] < fuel[DATES[0]]


def test_no_feasible_date_is_reported_explicitly():
    sweep = _sweep(["blocked"] * 4)
    assert sweep.selected is None
    assert "No departure date" in sweep.explanation


def test_option_summary_is_json_friendly():
    import json

    sweep = _sweep(["clear", "blocked", "clear", "clear"])
    json.dumps(sweep.to_dict())


def test_near_ties_within_tolerance_prefer_the_earliest_date():
    from antarctic_routing.routing.departure import select_departure

    class Opt:
        def __init__(self, day, fuel, hours):
            self.departure, self.expected_fuel, self.expected_hours, self.feasible = day, fuel, hours, True

    opts = [Opt(DATES[0], 1005.0, 40.0), Opt(DATES[1], 1000.0, 39.9), Opt(DATES[2], 900.0, 45.0)]
    assert select_departure(opts, fuel_tolerance=0.01).departure == DATES[2]   # clearly cheaper
    opts[2].expected_fuel = 999.0
    assert select_departure(opts, fuel_tolerance=0.01).departure == DATES[0]   # all within 1% -> earliest
