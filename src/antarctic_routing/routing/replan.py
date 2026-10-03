"""Stage 9 - replanning from the vessel's current state.

Every update re-evaluates the *remaining* part of the current route under the
newest joint scenarios and plans fresh candidates from the vessel's position.
The route is changed only for a stated reason:

* ``previous_route_exceeds_budget`` - the current route no longer meets the
  risk budget (switch if a feasible alternative exists, else report
  ``no_feasible_route`` for human decision);
* ``material_risk_reduction`` - a new route lowers the risk statistic by at
  least ``min_risk_reduction`` for at most ``max_extra_fuel`` extra fuel;
* ``material_fuel_saving`` - a new route saves at least ``min_fuel_saving``
  of fuel while meeting the budget;
* ``stale_input`` - inputs older than ``stale_hours`` (flagged, never silent).

Smaller differences keep the current route (hysteresis), and crew alerts are
raised only when the recommended path moves by more than ``deviation_km`` or
human judgement is required, to limit alert fatigue. Every decision can be
appended to a JSON-lines audit log.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from antarctic_routing.common.provenance import utc_now
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.routing.candidates import Candidate, is_feasible, plan_candidates
from antarctic_routing.routing.evaluate import RouteEvaluation, evaluate_route
from antarctic_routing.routing.fuel import VesselModel
from antarctic_routing.routing.graph import RoutingGrid
from antarctic_routing.routing.optimizer import Route
from antarctic_routing.synthetic import ScenarioSet


@dataclass(frozen=True)
class ReplanPolicy:
    risk_budget: float
    estimator: str = "wilson_upper"
    confidence: float = 0.95
    min_risk_reduction: float = 0.02
    max_extra_fuel: float = 0.05
    min_fuel_saving: float = 0.05
    deviation_km: float = 25.0
    stale_hours: float = 36.0


def remaining_route(route: Route, position: tuple[int, int]) -> Route:
    """Suffix of ``route`` from the vessel's cell, with times rebased to zero.

    The remaining distance is prorated by remaining planned time; exact
    distances come from :func:`evaluate_route`.
    """
    try:
        i = route.cells.index(tuple(position))
    except ValueError:
        raise ValueError(f"vessel position {position} is not on the current route") from None
    total = route.arrival_hours[-1] - route.arrival_hours[0]
    rest_h = route.arrival_hours[-1] - route.arrival_hours[i]
    share = rest_h / total if total > 0 else 0.0
    return Route(
        objective=f"{route.objective} (remaining)",
        cells=list(route.cells[i:]),
        arrival_hours=[h - route.arrival_hours[i] for h in route.arrival_hours[i:]],
        cost=float("nan"),
        distance_km=route.distance_km * share,
    )


def route_deviation_km(a: Sequence[tuple[int, int]], b: Sequence[tuple[int, int]], grid: PolarGrid) -> float:
    """Symmetric (discrete Hausdorff) distance between two cell paths, in projected km."""
    def xy(cells):
        r = np.array([c[0] for c in cells])
        c = np.array([c[1] for c in cells])
        return np.column_stack([grid.x[c], grid.y[r]]) / 1000.0

    pa, pb = xy(a), xy(b)
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


@dataclass
class ReplanDecision:
    action: str  # keep | switch | no_feasible_route
    triggers: list[str]
    explanation: str
    alert: bool
    previous_route: Route
    previous_evaluation: RouteEvaluation
    new_candidate: Candidate | None
    deviation_km: float
    extra: dict = field(default_factory=dict)

    @property
    def route(self) -> Route:
        """The route to follow after this decision."""
        if self.action == "switch" and self.new_candidate is not None:
            return self.new_candidate.route
        return self.previous_route

    def record(self, **context) -> dict:
        new = None
        if self.new_candidate is not None:
            new = {"labels": self.new_candidate.labels, **self.new_candidate.evaluation.summary(),
                   "cells": [list(c) for c in self.new_candidate.route.cells]}
        return {
            "logged_at": utc_now(), **context, "action": self.action, "triggers": self.triggers,
            "alert": self.alert, "explanation": self.explanation, "deviation_km": self.deviation_km,
            "previous": {**self.previous_evaluation.summary(),
                         "cells": [list(c) for c in self.previous_route.cells]},
            "new": new, **self.extra,
        }


def _stat(ev: RouteEvaluation, estimator: str) -> float:
    return ev.p_breach_upper if estimator == "wilson_upper" else ev.p_breach


def replan(
    world: ScenarioSet,
    position: tuple[int, int],
    current_route: Route,
    vessel: VesselModel,
    policy: ReplanPolicy,
    risk_weights: Sequence[float],
    connectivity: int = 16,
    n_scenario_routes: int = 2,
    depart_hours: float = 0.0,
    data_age_hours: float = 0.0,
    seed: int = 0,
    rgrid: RoutingGrid | None = None,
) -> ReplanDecision:
    rest = remaining_route(current_route, position)
    prev_ev = evaluate_route(rest, world, vessel, depart_hours=depart_hours, confidence=policy.confidence)
    prev_ok = is_feasible(prev_ev.breaches, prev_ev.n_scenarios, policy.risk_budget, policy.estimator,
                          policy.confidence)
    result = plan_candidates(
        world, vessel, tuple(position), rest.cells[-1], policy.risk_budget, risk_weights, policy.estimator,
        connectivity, n_scenario_routes, policy.confidence, seed, depart_hours=depart_hours, rgrid=rgrid,
    )
    triggers: list[str] = []
    if data_age_hours > policy.stale_hours:
        triggers.append("stale_input")
    if not prev_ok:
        triggers.append("previous_route_exceeds_budget")

    new = result.recommended
    prev_stat = _stat(prev_ev, policy.estimator)
    action = "keep"
    if new is None:
        if not prev_ok:
            action = "no_feasible_route"
    elif not prev_ok:
        action = "switch"
    else:
        new_stat = _stat(new.evaluation, policy.estimator)
        fuel_ratio = new.evaluation.expected_fuel / max(prev_ev.expected_fuel, 1e-9)
        if prev_stat - new_stat >= policy.min_risk_reduction and fuel_ratio <= 1 + policy.max_extra_fuel:
            action = "switch"
            triggers.append("material_risk_reduction")
        elif fuel_ratio <= 1 - policy.min_fuel_saving:
            action = "switch"
            triggers.append("material_fuel_saving")

    deviation = route_deviation_km(rest.cells, new.route.cells, world.grid) if action == "switch" else 0.0
    alert = action == "no_feasible_route" or "stale_input" in triggers or deviation >= policy.deviation_km

    if action == "keep":
        explanation = (f"Keep current route: remaining P(breach) {prev_ev.p_breach:.1%} "
                       f"(upper {prev_ev.p_breach_upper:.1%}) within the {policy.risk_budget:.0%} budget; "
                       "no alternative is materially better.")
    elif action == "switch":
        ev = new.evaluation
        explanation = (f"Switch route ({', '.join(triggers)}): remaining route P(breach) {prev_ev.p_breach:.1%} "
                       f"(upper {prev_ev.p_breach_upper:.1%}) -> new route {ev.p_breach:.1%} "
                       f"(upper {ev.p_breach_upper:.1%}); E[fuel] {prev_ev.expected_fuel:.0f} -> "
                       f"{ev.expected_fuel:.0f}; path moves up to {deviation:.0f} km.")
    else:
        explanation = (f"No route from the current position meets the {policy.risk_budget:.0%} risk budget "
                       f"(current route upper bound {prev_ev.p_breach_upper:.1%}). Human review required: "
                       "consider holding position, a different destination, or awaiting updated ice information.")
    if "stale_input" in triggers:
        explanation += f" Inputs are {data_age_hours:.0f} h old (limit {policy.stale_hours:.0f} h)."
    return ReplanDecision(action, triggers, explanation, alert, rest, prev_ev, new, deviation)


class AuditLog:
    """Append-only JSON-lines decision log."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str, sort_keys=True) + "\n")
