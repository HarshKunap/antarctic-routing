"""Stage 10.5 - sensitivity of recommendations to fuel and speed assumptions.

The relative fuel index d[1 + lambda g(C)] and the speed-in-ice factor
v(1 - kC) are documented assumptions, not measured vessel performance. This
analysis re-plans on the *same* joint scenarios for a grid of (lambda, k) and
reports, for each setting, whether the recommended route (path deviation from
the baseline recommendation) or departure date changes. A recommendation that
flips under plausible settings must be presented as uncertain.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from antarctic_routing.routing.candidates import plan_candidates
from antarctic_routing.routing.departure import plan_from_issue
from antarctic_routing.routing.fuel import VesselModel
from antarctic_routing.routing.graph import RoutingGrid
from antarctic_routing.routing.replan import route_deviation_km
from antarctic_routing.synthetic import ScenarioSet


def _recommend(world, vessel, origin, destination, risk_budget, risk_weights, estimator, connectivity, rgrid,
               offsets_days):
    if offsets_days is None:
        plan = plan_candidates(world, vessel, origin, destination, risk_budget, risk_weights, estimator,
                               connectivity, 0, rgrid=rgrid)
        return plan.recommended, None
    sweep = plan_from_issue(world, offsets_days, vessel, origin, destination, risk_budget, risk_weights,
                            estimator, connectivity, 0)
    if sweep.selected is None:
        return None, None
    return sweep.selected.plan.recommended, sweep.selected.lead_days


def run_sensitivity(
    world: ScenarioSet,
    vessel: VesselModel,
    origin: tuple[int, int],
    destination: tuple[int, int],
    lambdas: Sequence[float],
    ks: Sequence[float],
    risk_budget: float,
    risk_weights: Sequence[float],
    estimator: str = "wilson_upper",
    connectivity: int = 16,
    offsets_days: Sequence[int] | None = None,
    change_km: float = 10.0,
) -> dict:
    rgrid = RoutingGrid.build(world.grid, world.ocean, connectivity)
    base, base_lead = _recommend(world, vessel, origin, destination, risk_budget, risk_weights, estimator,
                                 connectivity, rgrid, offsets_days)
    rows = []
    for lam in lambdas:
        for k in ks:
            vm = replace(vessel, lam=float(lam), ice_k=float(k))
            rec, lead = _recommend(world, vm, origin, destination, risk_budget, risk_weights, estimator,
                                   connectivity, rgrid, offsets_days)
            if rec is None or base is None:
                dev = None
                changed = (rec is None) != (base is None)
            else:
                dev = route_deviation_km(base.route.cells, rec.route.cells, world.grid)
                changed = dev > change_km
            ev = rec.evaluation if rec else None
            rows.append({
                "lambda": float(lam), "k": float(k), "feasible": rec is not None,
                "route_changed": bool(changed), "deviation_km": dev,
                "departure_lead_days": lead, "departure_changed": lead != base_lead,
                "expected_fuel": ev.expected_fuel if ev else None,
                "expected_hours": ev.expected_hours if ev else None,
                "p_breach_upper": ev.p_breach_upper if ev else None,
            })
    n = len(rows)
    return {
        "baseline": {"lambda": vessel.lam, "k": vessel.ice_k, "departure_lead_days": base_lead,
                     "feasible": base is not None},
        "grid": rows,
        "stable_fraction": sum(not r["route_changed"] for r in rows) / n,
        "stable_departure_fraction": sum(not r["departure_changed"] for r in rows) / n,
        "change_threshold_km": change_km,
        "execution_mode": world.execution_mode,
    }
