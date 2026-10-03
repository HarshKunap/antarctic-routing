"""Stage 7 - candidate generation and risk-budgeted selection.

Route risk evaluated over joint scenarios is not additive along edges, so a
risk-constrained shortest path cannot be solved by Dijkstra directly. The planner
therefore works in two steps:

1. **Generate** diverse candidates with the time-dependent search: shortest
   distance, minimum time, minimum fuel, a sweep of risk-penalty weights, and
   routes optimal for individual sampled scenarios.
2. **Evaluate** every candidate on the *same* joint scenario set and solve

       min_R E[F_R]   subject to   P(B_R) <= r_max

   where P(B_R) is either the point estimate or (default) the upper Wilson
   bound, so a route is accepted only when the scenario sample *demonstrates*
   compliance.

If no candidate satisfies the budget the result is ``infeasible`` - the planner
never returns a recommendation it cannot support.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from antarctic_routing.config import wilson_upper_bound
from antarctic_routing.routing.evaluate import RouteEvaluation, evaluate_route
from antarctic_routing.routing.fuel import VesselModel
from antarctic_routing.routing.graph import RoutingGrid
from antarctic_routing.routing.optimizer import (
    NoRouteError,
    Objective,
    Route,
    layers_for_scenario,
    layers_from_scenarios,
    plan_route,
)
from antarctic_routing.synthetic import ScenarioSet


def is_feasible(breaches: int, n: int, budget: float, estimator: str, confidence: float = 0.95) -> bool:
    if estimator == "point":
        return breaches / n <= budget
    if estimator == "wilson_upper":
        return wilson_upper_bound(breaches, n, confidence) <= budget
    raise ValueError("estimator must be 'point' or 'wilson_upper'")


@dataclass
class Candidate:
    labels: list[str]
    route: Route
    evaluation: RouteEvaluation
    feasible: bool
    risk_statistic: float
    tags: list[str] = field(default_factory=list)


@dataclass
class PlanResult:
    status: str  # feasible | infeasible | no_route
    recommended: Candidate | None
    candidates: list[Candidate]
    explanation: str
    risk_budget: float
    estimator: str
    warnings: list[str] = field(default_factory=list)


def _objectives(risk_weights: Sequence[float]) -> list[Objective]:
    objs = [
        Objective("shortest_distance", w_distance=1.0),
        Objective("minimum_time", w_time=1.0),
        Objective("minimum_fuel", w_fuel=1.0),
    ]
    objs += [Objective(f"risk_weighted_{w:g}", w_fuel=1.0, w_risk=float(w)) for w in risk_weights if w > 0]
    return objs


def plan_candidates(
    world: ScenarioSet,
    vessel: VesselModel,
    origin: tuple[int, int],
    destination: tuple[int, int],
    risk_budget: float,
    risk_weights: Sequence[float],
    estimator: str = "wilson_upper",
    connectivity: int = 16,
    n_scenario_routes: int = 3,
    confidence: float = 0.95,
    seed: int = 0,
    depart_hours: float = 0.0,
    rgrid: RoutingGrid | None = None,
) -> PlanResult:
    rgrid = rgrid or RoutingGrid.build(world.grid, world.ocean, connectivity)
    warnings: list[str] = []
    routes: list[Route] = []

    mean_layers = layers_from_scenarios(world, vessel.tau)
    for obj in _objectives(risk_weights):
        try:
            routes.append(plan_route(rgrid, mean_layers, vessel, origin, destination, obj, depart_hours=depart_hours))
        except NoRouteError as exc:
            warnings.append(f"{obj.name}: {exc}")

    n_scen = min(n_scenario_routes, world.n_scenarios)
    if n_scen:
        w_r = max([w for w in risk_weights if w > 0], default=100.0)
        rng = np.random.default_rng(seed)
        for k in sorted(rng.choice(world.n_scenarios, n_scen, replace=False).tolist()):
            obj = Objective(f"scenario_{k}_optimal", w_fuel=1.0, w_risk=w_r)
            try:
                layers = layers_for_scenario(world, k, vessel.tau)
                routes.append(plan_route(rgrid, layers, vessel, origin, destination, obj, depart_hours=depart_hours))
            except NoRouteError as exc:
                warnings.append(f"{obj.name}: {exc}")

    if not routes:
        return PlanResult("no_route", None, [], "No navigable route exists between origin and destination.",
                          risk_budget, estimator, warnings)

    merged: dict[tuple, Candidate] = {}
    for route in routes:
        key = tuple(route.cells)
        if key in merged:
            merged[key].labels.append(route.objective)
            continue
        ev = evaluate_route(route, world, vessel, depart_hours=depart_hours, confidence=confidence)
        stat = ev.p_breach_upper if estimator == "wilson_upper" else ev.p_breach
        merged[key] = Candidate(
            labels=[route.objective], route=route, evaluation=ev,
            feasible=is_feasible(ev.breaches, ev.n_scenarios, risk_budget, estimator, confidence),
            risk_statistic=stat,
        )
    candidates = list(merged.values())

    lowest_risk = min(candidates, key=lambda c: (c.evaluation.p_breach, c.evaluation.expected_fuel))
    lowest_risk.tags.append("lowest_risk")
    feasible = [c for c in candidates if c.feasible]
    if not feasible:
        ev = lowest_risk.evaluation
        return PlanResult(
            "infeasible", None, candidates,
            (f"No candidate route satisfies the risk budget of {risk_budget:.1%} "
             f"({estimator}). Lowest-risk candidate: P(breach)={ev.p_breach:.1%}, "
             f"upper bound {ev.p_breach_upper:.1%} over {ev.n_scenarios} scenarios. "
             "Consider a later departure, a different destination, or a vessel with a higher ice capability."),
            risk_budget, estimator, warnings,
        )

    min(feasible, key=lambda c: (c.evaluation.expected_hours, c.evaluation.expected_fuel)).tags.append("fastest_feasible")
    min(feasible, key=lambda c: c.evaluation.distance_km).tags.append("shortest_feasible")
    best = min(feasible, key=lambda c: (c.evaluation.expected_fuel, c.evaluation.expected_hours))
    best.tags.append("lowest_fuel_feasible")
    ev = best.evaluation
    return PlanResult(
        "feasible", best, candidates,
        (f"Recommended the minimum expected-fuel route among {len(feasible)} of {len(candidates)} candidates "
         f"meeting the {risk_budget:.1%} risk budget ({estimator}): P(breach)={ev.p_breach:.1%} "
         f"(upper {ev.p_breach_upper:.1%}), E[time]={ev.expected_hours:.1f} h, E[fuel index]={ev.expected_fuel:.0f}."),
        risk_budget, estimator, warnings,
    )
