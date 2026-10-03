"""Stage 10.4 - historical replay backtest across seasons and start dates.

Each start date is replayed three ways and every voyage is sailed through the
*observed* ice:

* ``planner`` - this system: forecast-driven departure window, risk-budgeted
  routes, daily replanning (:func:`replay.run_replay`);
* ``naive`` - leave on the start date along the shortest route;
* ``ice_edge_buffer`` - a common operational rule of thumb: leave on the first
  day a shortest route exists that keeps at least ``buffer_km`` from the
  *currently observed* ice (C >= tau), and never replan.

Metrics: whether a voyage met observed hazardous ice, hours spent in it, cells
crossed, fuel index, sailing time, days waited and total elapsed time (the cost
of caution). For the planner, the mean predicted P(breach) at departure is
compared with the realised breach rate. A replay shows exposure to observed
conditions; it cannot show that an incident would or would not have occurred.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np
from scipy.ndimage import binary_dilation

from antarctic_routing.forecasting.scenarios import ForecastContext, grid_of
from antarctic_routing.preprocessing.climatology import season_of
from antarctic_routing.replay import run_replay, score_track
from antarctic_routing.routing.fuel import VesselModel
from antarctic_routing.routing.graph import RoutingGrid
from antarctic_routing.routing.optimizer import EnvironmentLayers, NoRouteError, Objective, plan_route
from antarctic_routing.routing.replan import ReplanPolicy

METHODS = ("planner", "naive", "ice_edge_buffer")


def buffer_baseline(
    ctx: ForecastContext,
    vessel: VesselModel,
    origin: tuple[int, int],
    destination: tuple[int, int],
    start: date,
    buffer_km: float,
    max_wait_days: int,
    connectivity: int = 16,
) -> dict:
    grid = grid_of(ctx.ds)
    ocean = ~ctx.ds["land_mask"].values.astype(bool)
    r = int(np.ceil(buffer_km * 1000.0 / grid.resolution_m))
    yy, xx = np.mgrid[-r: r + 1, -r: r + 1]
    disk = xx**2 + yy**2 <= r**2
    zeros = np.zeros((1, *ocean.shape))
    layers = EnvironmentLayers(zeros, zeros, zeros, zeros, 24.0)
    for k in range(max_wait_days + 1):
        day = start + timedelta(days=k)
        obs = ctx.ds["ice_concentration"].values[ctx.index_of(day)]
        ice = ocean & (np.nan_to_num(obs) >= vessel.tau)
        navigable = ocean & ~binary_dilation(ice, structure=disk)
        try:
            route = plan_route(RoutingGrid.build(grid, navigable, connectivity), layers, vessel, origin, destination,
                               Objective("shortest_distance", w_distance=1.0))
        except NoRouteError:
            continue
        return {"departure": day.isoformat(), "days_waited": k, "cells": route.cells}
    return {"departure": None, "days_waited": None, "cells": None}


def _summary(rows: list[dict]) -> dict:
    departed = [r for r in rows if r["departure"]]
    def mean(key):
        vals = [r[key] for r in departed if r[key] is not None and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else None
    return {
        "n": len(rows),
        "departed": len(departed),
        "breach_rate": float(np.mean([r["breached"] for r in departed])) if departed else None,
        "mean_hazard_hours": mean("hazard_hours"),
        "mean_observed_breach_cells": mean("observed_breach_cells"),
        "mean_fuel_index": mean("fuel_index"),
        "mean_sail_hours": mean("sail_hours"),
        "mean_days_waited": mean("days_waited"),
        "mean_elapsed_hours": mean("elapsed_hours"),
    }


def run_backtest(
    ctx: ForecastContext,
    vessel: VesselModel,
    origin: tuple[int, int],
    destination: tuple[int, int],
    starts: Sequence[date],
    window_days: int,
    max_wait_days: int,
    n_members: int,
    policy: ReplanPolicy,
    risk_weights: Sequence[float],
    voyage_days: int,
    buffer_km: float = 50.0,
    seed: int = 0,
    scenario_routes: int = 0,
    connectivity: int = 16,
) -> dict:
    for s in starts:
        if season_of(s, ctx.season_months) in set(ctx.train_seasons):
            raise ValueError(f"start {s} is in a training season; backtests must use held-out seasons")
    rows: list[dict] = []
    risk = []
    for i, start in enumerate(starts):
        season = season_of(start, ctx.season_months)
        rp = run_replay(ctx, vessel, origin, destination, start, window_days, max_wait_days, n_members, policy,
                        risk_weights, voyage_days, np.random.default_rng([seed, i]), connectivity=connectivity,
                        scenario_routes=scenario_routes, seed=seed)
        base = {"start": start.isoformat(), "season": season}
        if rp["departure"]:
            pl = rp["truth"]["planner"]
            waited = rp["days_waited"]
            rows.append({**base, "method": "planner", **pl, "days_waited": waited, "arrived": rp["arrived"],
                         "elapsed_hours": 24.0 * waited + pl["sail_hours"]})
            dep = next(r for r in rp["days"] if r.get("action") == "depart")
            risk.append((dep["p_breach"], dep["p_breach_upper"], pl["breached"]))
        else:
            rows.append({**base, "method": "planner", **_empty()})
        nv = rp["truth"]["naive"] if "truth" in rp else score_track(
            _naive_cells(ctx, vessel, origin, destination, connectivity), ctx, start, vessel)
        rows.append({**base, "method": "naive", **nv, "days_waited": 0, "arrived": True,
                     "elapsed_hours": nv["sail_hours"]})
        bf = buffer_baseline(ctx, vessel, origin, destination, start, buffer_km, max_wait_days, connectivity)
        if bf["departure"]:
            sc = score_track(bf["cells"], ctx, date.fromisoformat(bf["departure"]), vessel)
            rows.append({**base, "method": "ice_edge_buffer", **sc, "days_waited": bf["days_waited"],
                         "arrived": True, "elapsed_hours": 24.0 * bf["days_waited"] + sc["sail_hours"]})
        else:
            rows.append({**base, "method": "ice_edge_buffer", **_empty()})

    return {
        "starts": [s.isoformat() for s in starts],
        "buffer_km": buffer_km,
        "risk_budget": policy.risk_budget,
        "voyages": rows,
        "summary": {m: _summary([r for r in rows if r["method"] == m]) for m in METHODS},
        "planner_risk_check": {
            "n_departures": len(risk),
            "mean_predicted_p_breach": float(np.mean([r[0] for r in risk])) if risk else None,
            "mean_predicted_upper_bound": float(np.mean([r[1] for r in risk])) if risk else None,
            "realised_breach_rate": float(np.mean([r[2] for r in risk])) if risk else None,
        },
        "execution_mode": ctx.ds.attrs.get("execution_mode", "real"),
        "train_seasons": ctx.train_seasons,
    }


def _empty() -> dict:
    return {"departure": None, "days_waited": None, "hazard_hours": None, "observed_breach_cells": None,
            "breached": None, "fuel_index": None, "sail_hours": None, "elapsed_hours": None, "arrived": False}


def _naive_cells(ctx, vessel, origin, destination, connectivity):
    grid = grid_of(ctx.ds)
    ocean = ~ctx.ds["land_mask"].values.astype(bool)
    zeros = np.zeros((1, *ocean.shape))
    layers = EnvironmentLayers(zeros, zeros, zeros, zeros, 24.0)
    return plan_route(RoutingGrid.build(grid, ocean, connectivity), layers, vessel, origin, destination,
                      Objective("shortest_distance", w_distance=1.0)).cells
