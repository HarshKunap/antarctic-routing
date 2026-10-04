"""Real Historical Data endpoints - the real-data planner on past seasons, computed on request.

GET  /real/historical/status                 availability, coverage, hindsight-forcing disclosure, limitations
GET  /real/historical/dates                  issue dates the archive supports, with in-sample flags per season
POST /real/historical/routes[?wait=true]     risk-budgeted route for a departure on the issue date (job)
POST /real/historical/departures[?wait=true] departure window from one forecast issue (job)
POST /real/historical/voyages                create a voyage from a real route plan
POST /real/historical/voyages/{id}/replan    replan from a position on the route with a later forecast
POST /real/historical/replay[?wait=true]     day-by-day replay with icebergs, scored against observations (job)

The voyage history and export endpoints (``/voyages/{id}/history``, ``/export``) serve these voyages too.

Inputs are the verified files of ``config/real_historical.json`` under ``ANTROUTE_DATA_ROOT``; the model
and the scenario context load once, on the first computation. Parameters are the frozen ones (200 joint
scenarios, seed 42, calibrated drift), so nothing here is tunable per request. A missing input or
dependency answers 503 with the reason and a date the archive cannot support answers 422; nothing
synthetic is ever substituted. Every response is labelled "Real Historical Data" and carries the
hindsight-forcing disclosure.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from antarctic_routing import DISCLAIMER
from antarctic_routing.common.provenance import utc_now
from antarctic_routing.forecasting.scenarios import grid_of
from antarctic_routing.historical import (
    DATA_STATUS,
    LABEL,
    HistoricalArchive,
    HistoricalPlanner,
    HistoricalUnavailable,
    OutOfCoverage,
    provenance,
)
from antarctic_routing.ingestion.icebergs import in_grid_mask
from antarctic_routing.publish import UNAVAILABLE, grid_geometry, iceberg_tracks, to_percent
from antarctic_routing.routing.candidates import Candidate, plan_candidates
from antarctic_routing.routing.departure import plan_from_issue
from antarctic_routing.routing.replan import ReplanPolicy, replan

log = logging.getLogger("antarctic_routing.api")

LAND_LABEL = "Land (sea-ice product mask; not a navigational coastline)"
DATA_AGE_BASIS = ("Historical mode: a replan uses the sea-ice analysis of its issue date, assumed in hand that day "
                  "(operational product latency is not modelled); the USNIC list age is reported separately.")


class HistoricalRouteRequest(BaseModel):
    issue: date


class HistoricalWindowRequest(BaseModel):
    issue: date
    window_days: int = Field(default=14, ge=1, le=14)


class HistoricalReplanRequest(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    issued: date


class HistoricalReplayRequest(BaseModel):
    start: date
    max_wait_days: int = Field(default=7, ge=0, le=21)


class HistoricalService:
    """Loads the verified archive (cheap) and the U-Net scenario context (heavy) once, on demand."""

    def __init__(self, svc, data_root: str | None) -> None:
        self.svc = svc
        self.data_root = data_root
        self.spec_path = Path(os.environ.get("ANTROUTE_HISTORICAL_SPEC")
                              or svc.config_path.resolve().parent / "real_historical.json")
        self._archive: HistoricalArchive | None = None
        self._planner: HistoricalPlanner | None = None
        self._error: HistoricalUnavailable | None = None
        self._lock = threading.Lock()

    def _unavailable(self) -> HistoricalUnavailable | None:
        if not self.data_root:
            return HistoricalUnavailable(UNAVAILABLE, "no real-data archive configured (set ANTROUTE_DATA_ROOT)")
        if not self.spec_path.is_file():
            return HistoricalUnavailable("blocked", f"input list {self.spec_path.name} not found")
        return self._error

    def archive(self) -> HistoricalArchive:
        with self._lock:
            if self._archive is None and self._unavailable() is None:
                try:
                    self._archive = HistoricalArchive.load(self.data_root, self.spec_path,
                                                           self.svc.cfg.project.season_months)
                    log.info("Real Historical Data archive verified under %s", self.data_root)
                except HistoricalUnavailable as exc:
                    self._error = exc
                    log.error("Real Historical Data %s: %s", exc.status, exc.reason)
            err = self._unavailable()
        if err is not None:
            raise HTTPException(503, {"status": err.status, "reason": err.reason, "label": LABEL})
        return self._archive

    def planner(self) -> HistoricalPlanner:
        archive = self.archive()
        with self._lock:
            if self._planner is None and self._error is None:
                try:
                    self._planner = HistoricalPlanner(archive)
                    log.info("Real Historical Data forecast context built")
                except HistoricalUnavailable as exc:
                    self._error = exc
                    log.error("Real Historical Data %s: %s", exc.status, exc.reason)
            err = self._error
        if err is not None:
            raise HTTPException(503, {"status": err.status, "reason": err.reason, "label": LABEL})
        return self._planner

    def status(self) -> dict:
        out = {"label": LABEL, "execution_mode": "real", "data_status": DATA_STATUS}
        try:
            a = self.archive()
        except HTTPException as exc:
            return {**out, **exc.detail, "status": exc.detail["status"]}
        h = self.svc.horizon_days
        return {**out, "status": "available", "reason": None, "model_loaded": self._planner is not None,
                "hindsight_forcing": a.spec["hindsight_forcing"], "limitations": a.spec["limitations"],
                "parameters": a.params, "horizon_days": h, "window_days_max": a.params["window_days"],
                "coverage": {"route": a.coverage(1 + h), "window": a.coverage(a.params["window_days"] + h)}}


def _labels(a: HistoricalArchive) -> dict:
    return {"execution_mode": "real", "data_status": DATA_STATUS, "data_label": LABEL,
            "hindsight_forcing": a.spec["hindsight_forcing"], "disclaimer": DISCLAIMER}


def register(app: FastAPI, svc) -> HistoricalService:
    hs = HistoricalService(svc, os.environ.get("ANTROUTE_DATA_ROOT") or None)
    cfg = svc.cfg
    H = svc.horizon_days

    def coverage_or_422(issue: date, n_days: int) -> HistoricalArchive:
        a = hs.archive()
        try:
            a.check(issue, n_days)
        except OutOfCoverage as exc:
            raise HTTPException(422, {"status": "out_of_coverage", "reason": str(exc), "label": LABEL}) from None
        return a

    def tracks(planner: HistoricalPlanner, world, snap) -> list[dict]:
        if not snap.bergs:
            return []
        p = planner.archive.params
        return iceberg_tracks(world, snap.bergs, p["seed"], p["berg_radius_km"] * 1e3, **planner.drift_kwargs())

    def plan_at_issue(issue: date):
        planner = hs.planner()
        p = planner.archive.params
        world, snap = planner.world(issue, 1 + H)
        o, d = svc.endpoints(world.grid)
        r = cfg.routing
        plan = plan_candidates(world, svc.vessel, o, d, r.risk_budget, r.risk_weights, r.risk_estimator,
                               r.connectivity, p["scenario_routes"], r.confidence, p["seed"])
        return planner, world, snap, plan

    def route_payload(planner, world, snap, plan, issue: date, n_days: int) -> dict:
        a = planner.archive
        out = svc.plan_payload(world, plan)
        out.update(_labels(a), land_label=LAND_LABEL, icebergs=snap.summary(),
                   iceberg_tracks=tracks(planner, world, snap),
                   historical=provenance(a, cfg, issue, n_days, snap, planner))
        return out

    @app.get("/real/historical/status")
    def historical_status():
        return hs.status()

    @app.get("/real/historical/dates")
    def historical_dates():
        a = hs.archive()
        n_route, n_window = 1 + H, a.params["window_days"] + H
        return {**_labels(a), "horizon_days": H, "window_days_max": a.params["window_days"],
                "route_dates": [d.isoformat() for d in a.available(n_route)],
                "window_dates": [d.isoformat() for d in a.available(n_window)],
                "seasons": {"route": a.coverage(n_route), "window": a.coverage(n_window)}}

    @app.post("/real/historical/routes")
    def historical_routes(req: HistoricalRouteRequest, wait: bool = Query(False)):
        coverage_or_422(req.issue, 1 + H)

        def work():
            planner, world, snap, plan = plan_at_issue(req.issue)
            return route_payload(planner, world, snap, plan, req.issue, 1 + H)

        return svc.submit("historical_routes", work, wait)

    @app.post("/real/historical/departures")
    def historical_departures(req: HistoricalWindowRequest, wait: bool = Query(False)):
        n_days = req.window_days + H
        coverage_or_422(req.issue, n_days)

        def work():
            planner = hs.planner()
            a, p, r = planner.archive, planner.archive.params, cfg.routing
            world, snap = planner.world(req.issue, n_days)
            o, d = svc.endpoints(world.grid)
            sweep = plan_from_issue(world, list(range(req.window_days)), svc.vessel, o, d, r.risk_budget,
                                    r.risk_weights, r.risk_estimator, r.connectivity, p["scenario_routes"],
                                    r.confidence, p["seed"])
            sel = sweep.selected
            out = {**sweep.to_dict(), **_labels(a), "issue": req.issue.isoformat(), "window_days": req.window_days,
                   "horizon_days": H, "risk_budget": r.risk_budget, "n_scenarios": world.n_scenarios,
                   "layer_source": world.layer_source, "land_label": LAND_LABEL, "icebergs": snap.summary(),
                   "iceberg_tracks": tracks(planner, world, snap),
                   "historical": provenance(a, cfg, req.issue, n_days, snap, planner),
                   "selected_route": None, "map": None}
            g = world.grid
            out["origin_xy_km"] = [float(g.x[o[1]] / 1000), float(g.y[o[0]] / 1000)]
            out["destination_xy_km"] = [float(g.x[d[1]] / 1000), float(g.y[d[0]] / 1000)]
            if sel is not None and sel.plan.recommended is not None:
                mid = sel.lead_days * world.time_step_hours + sel.expected_hours / 2
                out["selected_route"] = svc.candidate_payload(sel.plan.recommended, world)
                out["map"] = svc.map_payload(world, svc.vessel.tau, 2 * mid)   # map_payload shows hours / 2
            else:
                out["map"] = svc.map_payload(world, svc.vessel.tau, 0.0)
            return out

        return svc.submit("historical_departures", work, wait)

    @app.post("/real/historical/voyages", status_code=201)
    def historical_create_voyage(req: HistoricalRouteRequest):
        coverage_or_422(req.issue, 1 + H)
        if len(svc.voyages) >= svc.max_voyages:
            raise HTTPException(503, "voyage store is full (in-memory, ANTROUTE_MAX_VOYAGES); restart to clear it")
        with svc.inline_slot():
            planner, world, snap, plan = plan_at_issue(req.issue)
            payload = route_payload(planner, world, snap, plan, req.issue, 1 + H)
        if plan.recommended is None:
            raise HTTPException(409, plan.explanation)
        from antarctic_routing.api.main import RouteContext

        vid = uuid.uuid4().hex[:12]
        cand = plan.recommended
        svc.voyages[vid] = {"mode": "historical", "issue": req.issue, "last_issue": req.issue, "candidate": cand,
                            "world": RouteContext.of(world), "data_labels": _labels(planner.archive), "events": [{
                                "event": "planned", "at": utc_now(), "departure": req.issue.isoformat(),
                                "issued": req.issue.isoformat(), "explanation": plan.explanation,
                                "usnic_list": snap.list_date.isoformat(), **cand.evaluation.summary()}]}
        return {"voyage_id": vid, "route": svc.candidate_payload(cand, world), "explanation": plan.explanation,
                **{k: payload[k] for k in ("map", "origin_xy_km", "destination_xy_km", "icebergs", "iceberg_tracks",
                                           "historical", "land_label")}, **_labels(planner.archive)}

    @app.post("/real/historical/voyages/{vid}/replan")
    def historical_replan(vid: str, req: HistoricalReplanRequest):
        if vid not in svc.voyages:
            raise HTTPException(404, "unknown voyage")
        v = svc.voyages[vid]
        if v.get("mode") != "historical":
            raise HTTPException(409, "this voyage was planned on controlled-synthetic data; replan it with "
                                     "/voyages/{id}/replan")
        if req.issued < v["last_issue"]:
            raise HTTPException(422, f"issued must not be before the last forecast used ({v['last_issue']})")
        coverage_or_422(req.issued, 1 + H)
        with svc.inline_slot():
            planner = hs.planner()
            p, r = planner.archive.params, cfg.routing
            world, snap = planner.world(req.issued, 1 + H)
            try:
                position = world.grid.cell_of(req.lat, req.lon)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from None
            policy = ReplanPolicy(r.risk_budget, r.risk_estimator, r.confidence)
            try:
                decision = replan(world, position, v["candidate"].route, svc.vessel, policy, r.risk_weights,
                                  r.connectivity, p["scenario_routes"], data_age_hours=0.0, seed=p["seed"])
            except ValueError as exc:
                raise HTTPException(400, f"{exc}; send a position on the current route") from None
        if decision.action == "switch":
            v["candidate"] = decision.new_candidate
        else:
            v["candidate"] = Candidate(v["candidate"].labels, decision.previous_route, decision.previous_evaluation,
                                       decision.action == "keep", decision.previous_evaluation.p_breach_upper)
        from antarctic_routing.api.main import RouteContext

        v["world"], v["last_issue"] = RouteContext.of(world), req.issued
        record = decision.record(issued=req.issued.isoformat(), event="replan", data_age_basis=DATA_AGE_BASIS,
                                 usnic_list=snap.list_date.isoformat(), usnic_age_days=snap.age_days)
        record.pop("previous", None)
        record.pop("new", None)
        v["events"].append(record)
        return {**record, "route": svc.candidate_payload(v["candidate"], world),
                "map": svc.map_payload(world, svc.vessel.tau, v["candidate"].evaluation.expected_hours),
                "icebergs": snap.summary(), "iceberg_tracks": tracks(planner, world, snap), "land_label": LAND_LABEL,
                "historical": provenance(planner.archive, cfg, req.issued, 1 + H, snap, planner),
                **_labels(planner.archive)}

    @app.post("/real/historical/replay")
    def historical_replay(req: HistoricalReplayRequest, wait: bool = Query(False)):
        a = hs.archive()
        window = a.params["window_days"]
        coverage_or_422(req.start, window + H)
        ok = 0
        while ok < req.max_wait_days and not a.problems(req.start + timedelta(days=ok + 1), window + H):
            ok += 1
        if ok < req.max_wait_days:
            raise HTTPException(422, {"status": "out_of_coverage", "label": LABEL,
                                      "reason": f"waiting {req.max_wait_days} days from {req.start} needs forecasts "
                                                f"past the archive's coverage; at most {ok} wait day(s) are possible"})

        def work():
            from antarctic_routing.replay import run_replay

            planner = hs.planner()
            p, r = planner.archive.params, cfg.routing
            grid = grid_of(planner.archive.ds)
            o, d = svc.endpoints(grid)
            policy = ReplanPolicy(r.risk_budget, r.risk_estimator, r.confidence)
            result = run_replay(planner.ctx, svc.vessel, o, d, req.start, window, req.max_wait_days, p["members"],
                                policy, r.risk_weights, H, np.random.default_rng(p["seed"]),
                                connectivity=r.connectivity, scenario_routes=p["scenario_routes"], seed=p["seed"],
                                hazard=planner.berg_hazard, truth_bergs=planner.observed_bergs,
                                berg_radius_m=p["berg_radius_km"] * 1e3)
            return replay_payload(planner, grid, result, req.start, window + H)

        return svc.submit("historical_replay", work, wait)

    def replay_payload(planner: HistoricalPlanner, grid, result: dict, start: date, n_days: int) -> dict:
        a = planner.archive
        xy = lambda cells: [[float(grid.x[c] / 1000), float(grid.y[rw] / 1000)] for rw, c in cells]   # noqa: E731
        ll = lambda cells: [[round(float(grid.lat2d[rw, c]), 5), round(float(grid.lon2d[rw, c]), 5)]  # noqa: E731
                            for rw, c in cells]
        out = {**result, **_labels(a), "land_label": LAND_LABEL,
               "historical": provenance(a, cfg, start, n_days, None, planner)}
        o, d = svc.endpoints(grid)
        out["origin_xy_km"], out["destination_xy_km"] = xy([o])[0], xy([d])[0]
        for key in ("sailed", "naive"):
            cells = result.get(f"{key}_cells")
            if cells:
                out[f"{key}_xy_km"], out[f"{key}_latlon"] = xy(cells), ll(cells)
        day = date.fromisoformat(result["departure"]) if result.get("departure") else start
        out["map"] = observed_map(a, grid, day)
        bergs = planner.observed_bergs(day)
        inside = in_grid_mask([b[1] for b in bergs], [b[2] for b in bergs], grid) if bergs else []
        out["observed_bergs"] = [{"id": b[0], "lat": b[1], "lon": b[2],
                                  "xy_km": [float(v) / 1000 for v in grid.to_xy(b[1], b[2])]}
                                 for b, m in zip(bergs, inside, strict=True) if m]
        return out

    def observed_map(a: HistoricalArchive, grid, day: date) -> dict:
        """Observed sea-ice concentration (percent) on ``day``, in the dashboard's map layout."""
        t = a.days.index(day)
        land = a.ds["land_mask"].values.astype(bool)
        conc = np.where(land, 0.0, a.ds["ice_concentration"].values[t])
        geo = grid_geometry(grid)
        return {**{k: geo[k] for k in ("nx", "ny", "res_km", "x0_km", "y0_km", "rotation_rad")}, "layer_day": 0,
                "observed_date": day.isoformat(), "land": land.astype(np.uint8).ravel().tolist(),
                "p_ice": to_percent(conc), "p_berg": None}

    return hs
