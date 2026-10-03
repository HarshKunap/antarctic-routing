"""Voyage replanning with triggers, hysteresis and an audit log (Stage 9)."""

import json

import numpy as np
import pytest

from _worlds import DIST, NX, NY, A, B, _vessel, _world
from antarctic_routing.routing.graph import RoutingGrid
from antarctic_routing.routing.optimizer import layers_from_scenarios, plan_route
from antarctic_routing.routing.replan import AuditLog, ReplanPolicy, remaining_route, replan, route_deviation_km

POLICY = ReplanPolicy(risk_budget=0.05)


def _straight(world):
    rg = RoutingGrid.build(world.grid, world.ocean, 16)
    return plan_route(rg, layers_from_scenarios(world, 0.15), _vessel(), A, B, DIST)


def test_remaining_route_starts_at_vessel_position_with_rebased_times():
    route = _straight(_world())
    rest = remaining_route(route, (5, 8))
    assert rest.cells[0] == (5, 8) and rest.cells[-1] == B
    assert rest.arrival_hours[0] == 0.0
    assert rest.distance_km < route.distance_km


def test_remaining_route_rejects_position_off_route():
    with pytest.raises(ValueError, match="not on"):
        remaining_route(_straight(_world()), (0, 0))


def test_route_deviation_km():
    world = _world()
    a = [(5, c) for c in range(NX)]
    b = [(7, c) for c in range(NX)]
    assert route_deviation_km(a, a, world.grid) == 0.0
    assert route_deviation_km(a, b, world.grid) == pytest.approx(20.0, rel=0.01)


def test_keep_previous_route_when_nothing_material_changes():
    world = _world(k=100)
    decision = replan(world, A, _straight(world), _vessel(), POLICY, risk_weights=[0, 100], n_scenario_routes=0)
    assert decision.action == "keep" and decision.alert is False
    assert decision.triggers == []


def test_switch_when_new_ice_puts_previous_route_over_budget():
    world0 = _world(k=100)
    previous = _straight(world0)
    conc = np.zeros((100, 3, NY, NX))
    conc[:, :, 3:8, 9:12] = 0.9                       # new ice on the planned line
    world1 = _world(k=100, conc=conc)
    decision = replan(world1, A, previous, _vessel(), POLICY, risk_weights=[0, 100], n_scenario_routes=0)
    assert decision.action == "switch"
    assert "previous_route_exceeds_budget" in decision.triggers
    assert decision.previous_evaluation.p_breach == 1.0
    assert decision.new_candidate.evaluation.p_breach == 0.0
    assert decision.deviation_km >= POLICY.deviation_km and decision.alert is True


def test_no_feasible_route_is_reported_not_hidden():
    conc = np.zeros((100, 3, NY, NX))
    conc[:, :, :, 9:12] = 0.9
    world = _world(k=100, conc=conc)
    decision = replan(world, A, _straight(_world()), _vessel(), POLICY, risk_weights=[0, 100], n_scenario_routes=0)
    assert decision.action == "no_feasible_route" and decision.alert is True
    assert "human" in decision.explanation.lower()


def test_stale_inputs_are_flagged():
    world = _world(k=100)
    decision = replan(world, A, _straight(world), _vessel(), POLICY, risk_weights=[0], n_scenario_routes=0,
                      data_age_hours=72)
    assert "stale_input" in decision.triggers and decision.alert is True


def test_audit_log_appends_json_lines(tmp_path):
    world = _world(k=100)
    decision = replan(world, A, _straight(world), _vessel(), POLICY, risk_weights=[0], n_scenario_routes=0)
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append(decision.record(day="2026-12-01"))
    log.append(decision.record(day="2026-12-02"))
    lines = [json.loads(x) for x in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert [x["day"] for x in lines] == ["2026-12-01", "2026-12-02"]
    assert {"action", "triggers", "explanation", "previous", "new", "logged_at"} <= lines[0].keys()
