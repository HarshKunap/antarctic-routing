"""Behavioural tests for the time-dependent router on small hand-built worlds."""

import numpy as np
import pytest

from antarctic_routing.routing.candidates import is_feasible, plan_candidates
from antarctic_routing.routing.evaluate import evaluate_route
from antarctic_routing.routing.graph import RoutingGrid
from antarctic_routing.routing.optimizer import (
    NoRouteError,
    Objective,
    layers_from_scenarios,
    plan_route,
)

from _worlds import A, B, DIST, NX, NY, TIME, _grid, _vessel, _world


def _plan(world, objective, start=A, goal=B, vessel=None, connectivity=16, **kw):
    rg = RoutingGrid.build(world.grid, world.ocean, connectivity)
    layers = layers_from_scenarios(world, (vessel or _vessel()).tau)
    return plan_route(rg, layers, vessel or _vessel(), start, goal, objective, **kw)


# ---------------------------------------------------------------- geometry
def test_open_water_shortest_route_is_straight():
    route = _plan(_world(), DIST)
    assert route.cells[0] == A and route.cells[-1] == B
    assert all(r == 5 for r, _ in route.cells)
    assert route.distance_km == pytest.approx(200, rel=0.1)  # ~20 cells at ~10 km


def test_route_passes_through_gap_in_land_wall():
    land = np.zeros((NY, NX), bool)
    land[:, 10] = True
    land[9, 10] = False
    route = _plan(_world(land=land), DIST)
    assert (9, 10) in route.cells
    assert not any(land[c] for c in route.cells)


def test_diagonal_moves_cannot_cut_land_corners():
    land = np.zeros((NY, NX), bool)
    land[4, 5] = True
    rg = RoutingGrid.build(_grid(), ~land, 8)
    assert not rg.edge_valid((4, 4), (5, 5))   # (4,5) is land
    assert rg.edge_valid((6, 6), (7, 7))


def test_sixteen_connectivity_is_never_longer_than_eight():
    world = _world()
    d8 = _plan(world, DIST, start=(0, 0), goal=(7, 20), connectivity=8).distance_km
    d16 = _plan(world, DIST, start=(0, 0), goal=(7, 20), connectivity=16).distance_km
    assert d16 <= d8 + 1e-9


def test_unreachable_destination_raises():
    land = np.zeros((NY, NX), bool)
    land[3:8, 17] = land[3:8, 19] = land[3, 17:20] = land[7, 17:20] = True
    with pytest.raises(NoRouteError):
        _plan(_world(land=land), DIST, goal=(5, 18))


# ------------------------------------------------------------ physics/time
def test_following_current_is_faster_than_opposing():
    world = _world(cx=1.0)  # +1 m/s along grid +x (A -> B)
    there = _plan(world, TIME)
    back = _plan(world, TIME, start=B, goal=A)
    assert there.arrival_hours[-1] < back.arrival_hours[-1]
    d = there.distance_km
    assert there.arrival_hours[-1] == pytest.approx(d / (12 * 1.852 + 3.6), rel=0.02)


def test_breach_depends_on_arrival_time_not_just_location():
    conc = np.zeros((4, 3, NY, NX))
    conc[:, 1:, :, 10] = 0.9  # ice band appears only from 24 h onwards
    world = _world(conc=conc)
    route = _plan(world, DIST)
    fast = evaluate_route(route, world, _vessel(12.0))
    slow = evaluate_route(route, world, _vessel(2.0, min_kmh=1.0))
    assert fast.breaches == 0          # crosses column 10 after ~4.5 h
    assert slow.breaches == 4          # arrives after 24 h, every scenario breaches
    assert slow.expected_hours > 24


def test_evaluation_counts_scenarios_independently():
    conc = np.zeros((4, 3, NY, NX))
    conc[:2, :, 5, 10] = 0.5  # only scenarios 0 and 1 have ice on the straight line
    world = _world(conc=conc)
    ev = evaluate_route(_plan(world, DIST), world, _vessel())
    assert ev.breaches == 2 and ev.p_breach == pytest.approx(0.5)


def test_astar_heuristic_does_not_change_optimal_cost():
    rng = np.random.default_rng(0)
    conc = np.repeat(rng.uniform(0, 0.4, (1, 1, NY, NX)), 4, axis=0).repeat(3, axis=1)
    world = _world(conc=conc, cx=0.3)
    obj = Objective("mixed", w_time=1.0, w_fuel=0.5, w_risk=50.0)
    with_h = _plan(world, obj, use_heuristic=True)
    without = _plan(world, obj, use_heuristic=False)
    assert with_h.cost == pytest.approx(without.cost, rel=1e-9)


# ------------------------------------------------------- risk & selection
def _blob_world(rows=slice(2, 9), k=100):
    conc = np.zeros((k, 3, NY, NX))
    conc[:, :, rows, 8:13] = 0.9
    return _world(k=k, conc=conc)


def test_risk_weighted_route_avoids_ice_that_distance_route_crosses():
    world = _blob_world()
    vessel = _vessel()
    dist_route = _plan(world, DIST)
    safe_route = _plan(world, Objective("risk", w_fuel=1.0, w_risk=100.0))
    assert evaluate_route(dist_route, world, vessel).p_breach == 1.0
    assert evaluate_route(safe_route, world, vessel).p_breach == 0.0
    assert safe_route.distance_km > dist_route.distance_km


@pytest.mark.parametrize(
    "breaches,n,estimator,expected",
    [(0, 100, "wilson_upper", True), (2, 100, "point", True),
     (2, 100, "wilson_upper", False), (6, 100, "point", False)],
)
def test_feasibility_respects_estimator(breaches, n, estimator, expected):
    assert is_feasible(breaches, n, 0.05, estimator, 0.95) is expected


def test_planner_recommends_feasible_min_fuel_route():
    world = _blob_world()
    result = plan_candidates(world, _vessel(), A, B, risk_budget=0.05, risk_weights=[0, 100],
                             estimator="wilson_upper", connectivity=16, n_scenario_routes=2)
    assert result.status == "feasible"
    rec = result.recommended
    assert rec.evaluation.breaches == 0
    feasible = [c for c in result.candidates if c.feasible]
    assert rec.evaluation.expected_fuel == min(c.evaluation.expected_fuel for c in feasible)
    labels = {lab for c in result.candidates for lab in c.labels}
    assert {"shortest_distance", "minimum_time", "minimum_fuel"} <= labels
    assert "lowest_fuel_feasible" in rec.tags


def test_planner_reports_infeasible_instead_of_recommending():
    world = _blob_world(rows=slice(0, NY))  # ice band spans the whole domain
    result = plan_candidates(world, _vessel(), A, B, risk_budget=0.05, risk_weights=[0, 100],
                             estimator="wilson_upper", connectivity=16, n_scenario_routes=0)
    assert result.status == "infeasible"
    assert result.recommended is None
    assert "risk budget" in result.explanation


def test_iced_destination_is_diagnosed_as_unavoidable():
    conc = np.zeros((100, 3, NY, NX))
    conc[:30, :, 4:7, 19:21] = 0.9  # destination area iced in 30% of scenarios
    world = _world(k=100, conc=conc)
    result = plan_candidates(world, _vessel(), A, B, risk_budget=0.05, risk_weights=[0, 100],
                             n_scenario_routes=0)
    assert result.status == "infeasible"
    assert result.candidates[0].evaluation.destination_breach_prob == pytest.approx(0.3)
    assert "destination itself" in result.explanation
