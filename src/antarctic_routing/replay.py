"""Stage 9.3 / 10.4 - historical replay of a voyage, one day at a time.

For each day only information available on that day is used:

* **In port** - issue a forecast (:class:`ForecastContext`) and plan the
  departure window from it; depart only when the selected departure is today.
* **At sea** - advance the vessel 24 h along its route through the *observed*
  (truth) ice, then issue the next day's forecast and :func:`replan` from the
  new position.

The sailed track is then scored against the observed ice and compared with a
naive plan (leave on the first day along the shortest route). A replay shows
whether the route met hazardous *observed* conditions; it cannot prove that a
real ship would or would not have had an incident.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from antarctic_routing import DISCLAIMER
from antarctic_routing.forecasting.scenarios import ForecastContext, grid_of
from antarctic_routing.preprocessing.climatology import season_of
from antarctic_routing.routing.departure import plan_from_issue
from antarctic_routing.routing.evaluate import evaluate_route
from antarctic_routing.routing.fuel import VesselModel
from antarctic_routing.routing.graph import RoutingGrid
from antarctic_routing.routing.optimizer import Objective, Route, layers_from_scenarios, plan_route
from antarctic_routing.routing.replan import AuditLog, ReplanPolicy, replan
from antarctic_routing.synthetic import ScenarioSet


def _truth_world(ctx: ForecastContext, day: date, n_days: int) -> ScenarioSet:
    """Observed ice from ``day`` onward as a single 'scenario' (clamped to the season)."""
    t = ctx.index_of(day)
    days = ctx.days
    season = season_of(day, ctx.season_months)
    n = 1
    while n < n_days and t + n < len(days) and season_of(days[t + n], ctx.season_months) == season \
            and (days[t + n] - days[t]).days == n:
        n += 1
    conc = ctx.ds["ice_concentration"].values[t: t + n][None].astype(np.float32)
    shape = conc.shape[2:]
    daily = ctx.current_layers(day, n)      # daily currents by exact date; 2-D currents repeat as before
    cx, cy = ctx.currents if daily is None else daily
    return ScenarioSet(
        grid=grid_of(ctx.ds), start=datetime(day.year, day.month, day.day), time_step_hours=24.0,
        land=ctx.ds["land_mask"].values.astype(bool), conc=conc,
        current_x=np.broadcast_to(cx, (n, *shape)).copy(), current_y=np.broadcast_to(cy, (n, *shape)).copy(),
        execution_mode=ctx.ds.attrs.get("execution_mode", "real"), description=f"observed ice from {day}",
        layer_source=["observed"] * n,
    )


def _sail(route: Route, ctx: ForecastContext, day: date, vessel: VesselModel, hours: float = 24.0):
    """Advance ``hours`` along ``route`` through observed ice; return (cells sailed, finished)."""
    ev = evaluate_route(route, _truth_world(ctx, day, 4), vessel)
    arrivals = ev.scenario_arrival_hours[0]
    last = 0
    for i, h in enumerate(arrivals):
        if np.isfinite(h) and h <= hours:
            last = i
    return route.cells[1: last + 1], last == len(route.cells) - 1


def score_track(cells: list[tuple[int, int]], ctx: ForecastContext, day: date, vessel: VesselModel) -> dict:
    """Sail ``cells`` from ``day`` through observed ice and measure the exposure."""
    route = Route("sailed", cells, [0.0] * len(cells), float("nan"), 0.0)
    ev = evaluate_route(route, _truth_world(ctx, day, 6), vessel)
    arrivals = ev.scenario_arrival_hours[0]
    hit = [p > 0 for p in ev.segment_breach_prob]
    hazard_hours = sum(arrivals[i] - arrivals[i - 1] for i in range(1, len(cells))
                       if hit[i] and np.isfinite(arrivals[i]))
    return {
        "departure": day.isoformat(),
        "hours": ev.expected_hours,
        "sail_hours": ev.expected_hours,
        "fuel_index": ev.expected_fuel,
        "distance_km": ev.distance_km,
        "observed_breach_cells": int(sum(hit)),
        "hazard_hours": float(hazard_hours),
        "breached": bool(any(hit)),
    }


def run_replay(
    ctx: ForecastContext,
    vessel: VesselModel,
    origin: tuple[int, int],
    destination: tuple[int, int],
    start: date,
    window_days: int,
    max_wait_days: int,
    n_members: int,
    policy: ReplanPolicy,
    risk_weights: Sequence[float],
    voyage_days: int,
    rng: np.random.Generator,
    audit_path: str | Path | None = None,
    trust_horizon_days: int | None = None,
    require_trusted: bool = False,
    connectivity: int = 16,
    scenario_routes: int = 1,
    seed: int = 0,
    max_sea_days: int = 10,
) -> dict:
    audit = AuditLog(audit_path) if audit_path else None
    records: list[dict] = []

    def log(rec: dict) -> None:
        records.append(rec)
        if audit:
            audit.append(rec)

    grid = grid_of(ctx.ds)
    day, route, departed = start, None, None
    for _ in range(max_wait_days + 1):
        world = ctx.scenarios(day, window_days + voyage_days, n_members, rng)
        sweep = plan_from_issue(world, list(range(window_days)), vessel, origin, destination, policy.risk_budget,
                                risk_weights, policy.estimator, connectivity, scenario_routes, policy.confidence,
                                seed, trust_horizon_days=trust_horizon_days, require_trusted=require_trusted)
        sel = sweep.selected
        go = sel is not None and sel.lead_days == 0
        log({"day": day.isoformat(), "phase": "port", "action": "depart" if go else "wait",
             "recommended_departure": sel.departure.isoformat() if sel else None,
             "p_breach": sel.p_breach if sel else None, "p_breach_upper": sel.p_breach_upper if sel else None,
             "explanation": sweep.explanation})
        if go:
            route, departed = sel.plan.recommended.route, day
            break
        day += timedelta(days=1)

    result = {"start": start.isoformat(), "departure": departed.isoformat() if departed else None,
              "days_waited": (departed - start).days if departed else None, "arrived": False,
              "execution_mode": ctx.ds.attrs.get("execution_mode", "real"), "policy": asdict(policy),
              "days": records, "disclaimer": DISCLAIMER}
    if route is None:
        return result

    sailed = [route.cells[0]]
    rgrid = RoutingGrid.build(grid, ~ctx.ds["land_mask"].values.astype(bool), connectivity)
    for _ in range(max_sea_days):
        path, finished = _sail(route, ctx, day, vessel)
        sailed += path
        position = sailed[-1]
        day += timedelta(days=1)
        lat, lon = grid.cell_latlon(*position)
        if finished:
            result["arrived"] = True
            log({"day": day.isoformat(), "phase": "arrived", "action": "arrived",
                 "position": [round(lat, 4), round(lon, 4)], "explanation": "Destination reached."})
            break
        world = ctx.scenarios(day, voyage_days, n_members, rng)
        decision = replan(world, position, route, vessel, policy, risk_weights, connectivity, scenario_routes,
                          seed=seed, rgrid=rgrid)
        route = decision.route
        log(decision.record(day=day.isoformat(), phase="at_sea", position=[round(lat, 4), round(lon, 4)]))

    naive_world = ctx.scenarios(start, voyage_days, 1, np.random.default_rng(seed))
    naive = plan_route(rgrid, layers_from_scenarios(naive_world, vessel.tau), vessel, origin, destination,
                       Objective("shortest_distance", w_distance=1.0))
    result["arrival_day"] = day.isoformat() if result["arrived"] else None
    result["sailed_cells"] = [list(c) for c in sailed]
    result["naive_cells"] = [list(c) for c in naive.cells]
    result["truth"] = {"planner": score_track(sailed, ctx, departed, vessel),
                       "naive": score_track(naive.cells, ctx, start, vessel)}
    return result
