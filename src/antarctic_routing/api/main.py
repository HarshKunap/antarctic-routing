"""Stage 11 - FastAPI backend.

Endpoints
---------
GET  /health                       service status, version, disclaimer
GET  /config                       validated scenario configuration
POST /routes[?wait=true]           plan risk-budgeted routes (job)
POST /departures[?wait=true]       sweep a departure window (job)
GET  /jobs/{id}                    job status and result
POST /voyages                      create a voyage from a route plan
POST /voyages/{id}/replan          replan from the vessel position
GET  /voyages/{id}/history         planned route and every replan decision
GET  /voyages/{id}/export          GeoJSON or CSV of the current route
GET  /                             dashboard (static)

Long computations run as background jobs so HTTP requests are never held open
for a whole ensemble; ``?wait=true`` runs them inline (scripts and tests).
Responses carry geometry and summaries, never full scientific arrays.
"""

from __future__ import annotations

import csv
import io
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from antarctic_routing import DISCLAIMER, __version__
from antarctic_routing.common.provenance import utc_now
from antarctic_routing.config import ProjectConfig, load_config, min_scenarios_for_budget
from antarctic_routing.export import route_to_geojson
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.routing.candidates import Candidate, plan_candidates
from antarctic_routing.routing.departure import sweep_departures
from antarctic_routing.routing.fuel import VesselModel
from antarctic_routing.routing.hazard import exceedance_probability
from antarctic_routing.routing.replan import ReplanPolicy, replan
from antarctic_routing.synthetic import ScenarioSet, generate_synthetic

DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"


class Berg(BaseModel):
    id: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class RouteRequest(BaseModel):
    departure: date
    scenarios: int | None = Field(default=None, ge=1, le=1000)
    resolution_km: float | None = Field(default=None, gt=0, le=100)
    seed: int = 42
    scenario_routes: int = Field(default=2, ge=0, le=10)
    icebergs: list[Berg] = []
    berg_radius_km: float = Field(default=10.0, ge=0)


class DepartureRequest(BaseModel):
    start: date
    end: date
    step_days: int = Field(default=1, ge=1)
    scenarios: int | None = Field(default=None, ge=1, le=1000)
    resolution_km: float | None = Field(default=None, gt=0, le=100)
    seed: int = 42


class ReplanRequest(BaseModel):
    lat: float
    lon: float
    issued: date
    scenarios: int | None = Field(default=None, ge=1, le=1000)
    data_age_hours: float = Field(default=0.0, ge=0)
    seed: int = 42


def _horizon_days(cfg: ProjectConfig) -> int:
    from antarctic_routing.cli import _horizon_days as h

    return h(cfg)


class Service:
    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.cfg = load_config(self.config_path)
        self.vessel = VesselModel.from_config(self.cfg)
        self._grids: dict[float, PolarGrid] = {}
        self.jobs: dict[str, dict] = {}
        self.voyages: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=2)

    # -------------------------------------------------------------- helpers
    def grid(self, resolution_km: float | None) -> PolarGrid:
        res = float(resolution_km or self.cfg.grid.resolution_km)
        if res not in self._grids:
            self._grids[res] = PolarGrid.from_domain(self.cfg.domain, res)
        return self._grids[res]

    def check_scenarios(self, n: int | None) -> int:
        n = n or self.cfg.forecast.route_scenarios
        r = self.cfg.routing
        if r.risk_estimator == "wilson_upper":
            need = min_scenarios_for_budget(r.risk_budget, r.confidence)
            if n < need:
                raise HTTPException(422, f"{n} scenarios cannot certify the {r.risk_budget:.0%} risk budget; "
                                         f"at least {need} are required")
        return n

    def world(self, start: date, n: int, resolution_km: float | None, seed: int,
              bergs: list[Berg] = (), berg_radius_km: float = 10.0) -> ScenarioSet:
        world = generate_synthetic(self.cfg, self.grid(resolution_km), start, _horizon_days(self.cfg), n, seed=seed)
        if bergs:
            from antarctic_routing.iceberg.drift import add_iceberg_hazard

            world = add_iceberg_hazard(world, [(b.id, b.lat, b.lon) for b in bergs],
                                       rng=np.random.default_rng(seed), radius_m=berg_radius_km * 1e3)
        return world

    def endpoints(self, grid: PolarGrid):
        o, d = self.cfg.route.origin, self.cfg.route.destination
        return grid.cell_of(o.lat, o.lon), grid.cell_of(d.lat, d.lon)

    # --------------------------------------------------------- serialisers
    @staticmethod
    def map_payload(world: ScenarioSet, tau: float, hours: float) -> dict:
        g = world.grid
        t = world.time_index(hours / 2 if np.isfinite(hours) else 0.0)
        p_ice = np.nan_to_num(exceedance_probability(world.conc[:, t], tau))
        p_berg = world.berg[:, t].mean(axis=0) if world.berg is not None else None
        res_km = g.resolution_m / 1000.0
        return {
            "nx": int(g.x.size), "ny": int(g.y.size), "res_km": res_km,
            "x0_km": float(g.x[0] / 1000.0 - res_km / 2), "y0_km": float(g.y[0] / 1000.0 - res_km / 2),
            "rotation_rad": float(np.deg2rad(np.median(g.lon2d))),
            "layer_day": t,
            "land": world.land.astype(np.uint8).ravel().tolist(),
            "p_ice": np.round(p_ice * 100).astype(int).ravel().tolist(),
            "p_berg": np.round(p_berg * 100).astype(int).ravel().tolist() if p_berg is not None else None,
        }

    @staticmethod
    def candidate_payload(c: Candidate, world: ScenarioSet) -> dict:
        g = world.grid
        return {
            "labels": c.labels, "tags": c.tags, "feasible": c.feasible,
            **c.evaluation.summary(),
            "xy_km": [[float(g.x[col] / 1000.0), float(g.y[row] / 1000.0)] for row, col in c.route.cells],
            "latlon": [[round(float(g.lat2d[row, col]), 5), round(float(g.lon2d[row, col]), 5)]
                       for row, col in c.route.cells],
            "segment_breach_prob": c.evaluation.segment_breach_prob,
        }

    def plan_payload(self, world: ScenarioSet, plan, bergs=()) -> dict:
        g = world.grid
        o, d = self.endpoints(g)
        rec_hours = plan.recommended.evaluation.expected_hours if plan.recommended else 24.0
        return {
            "status": plan.status, "explanation": plan.explanation, "risk_budget": plan.risk_budget,
            "estimator": plan.estimator, "departure_utc": world.start.isoformat() + "Z",
            "n_scenarios": world.n_scenarios, "execution_mode": world.execution_mode,
            "data_description": world.description,
            "recommended_index": next((i for i, c in enumerate(plan.candidates) if c is plan.recommended), None),
            "candidates": [self.candidate_payload(c, world) for c in plan.candidates],
            "origin_xy_km": [float(g.x[o[1]] / 1000), float(g.y[o[0]] / 1000)],
            "destination_xy_km": [float(g.x[d[1]] / 1000), float(g.y[d[0]] / 1000)],
            "icebergs": [b.model_dump() for b in bergs],
            "map": self.map_payload(world, self.vessel.tau, rec_hours),
            "disclaimer": DISCLAIMER,
        }

    # ---------------------------------------------------------------- jobs
    def submit(self, kind: str, fn, wait: bool):
        job_id = uuid.uuid4().hex[:12]
        job = {"job_id": job_id, "kind": kind, "status": "queued", "created_at": utc_now(), "result": None,
               "error": None}
        with self.lock:
            self.jobs[job_id] = job

        def run():
            job["status"] = "running"
            try:
                job["result"] = fn()
                job["status"] = "done"
            except Exception as exc:  # noqa: BLE001 - surfaced to the client as a failed job
                job["status"], job["error"] = "failed", f"{type(exc).__name__}: {exc}"
            job["finished_at"] = utc_now()

        if wait:
            run()
            return JSONResponse(job, status_code=200)
        self.pool.submit(run)
        return JSONResponse({"job_id": job_id, "status": "queued"}, status_code=202)


def create_app(config_path: str | Path | None = None) -> FastAPI:
    svc = Service(config_path or os.environ.get("ANTROUTE_CONFIG", "config/config.yaml"))
    app = FastAPI(title="Antarctic Vessel Routing & Ice-Risk API", version=__version__,
                  description=DISCLAIMER)
    app.state.service = svc

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "time": utc_now(), "disclaimer": DISCLAIMER,
                "data_modes": ["controlled_synthetic"], "jobs": len(svc.jobs), "voyages": len(svc.voyages)}

    @app.get("/config")
    def config():
        r = svc.cfg.routing
        return {"config": svc.cfg.model_dump(mode="json"),
                "min_scenarios_for_budget": min_scenarios_for_budget(r.risk_budget, r.confidence),
                "config_file": str(svc.config_path)}

    @app.post("/routes")
    def routes(req: RouteRequest, wait: bool = Query(False)):
        n = svc.check_scenarios(req.scenarios)

        def work():
            world = svc.world(req.departure, n, req.resolution_km, req.seed, req.icebergs, req.berg_radius_km)
            o, d = svc.endpoints(world.grid)
            r = svc.cfg.routing
            plan = plan_candidates(world, svc.vessel, o, d, r.risk_budget, r.risk_weights, r.risk_estimator,
                                   r.connectivity, req.scenario_routes, r.confidence, req.seed)
            return svc.plan_payload(world, plan, req.icebergs)

        return svc.submit("routes", work, wait)

    @app.post("/departures")
    def departures(req: DepartureRequest, wait: bool = Query(False)):
        n = svc.check_scenarios(req.scenarios)
        if req.end < req.start:
            raise HTTPException(422, "end must not be before start")

        def work():
            dates, cur = [], req.start
            while cur <= req.end:
                dates.append(cur)
                cur += timedelta(days=req.step_days)
            grid = svc.grid(req.resolution_km)
            o, d = svc.endpoints(grid)
            r = svc.cfg.routing
            sweep = sweep_departures(dates, lambda day: svc.world(day, n, req.resolution_km, req.seed), svc.vessel,
                                     o, d, r.risk_budget, r.risk_weights, r.risk_estimator, r.connectivity, 0,
                                     r.confidence, req.seed)
            return {**sweep.to_dict(), "execution_mode": "controlled_synthetic", "disclaimer": DISCLAIMER}

        return svc.submit("departures", work, wait)

    @app.get("/jobs/{job_id}")
    def job(job_id: str):
        if job_id not in svc.jobs:
            raise HTTPException(404, "unknown job")
        return svc.jobs[job_id]

    @app.post("/voyages", status_code=201)
    def create_voyage(req: RouteRequest):
        n = svc.check_scenarios(req.scenarios)
        world = svc.world(req.departure, n, req.resolution_km, req.seed, req.icebergs, req.berg_radius_km)
        o, d = svc.endpoints(world.grid)
        r = svc.cfg.routing
        plan = plan_candidates(world, svc.vessel, o, d, r.risk_budget, r.risk_weights, r.risk_estimator,
                               r.connectivity, req.scenario_routes, r.confidence, req.seed)
        if plan.recommended is None:
            raise HTTPException(409, plan.explanation)
        vid = uuid.uuid4().hex[:12]
        cand = plan.recommended
        svc.voyages[vid] = {"request": req, "candidate": cand, "world": world, "events": [{
            "event": "planned", "at": utc_now(), "departure": req.departure.isoformat(),
            "explanation": plan.explanation, **cand.evaluation.summary()}]}
        return {"voyage_id": vid, "route": svc.candidate_payload(cand, world), "explanation": plan.explanation,
                "disclaimer": DISCLAIMER}

    def _voyage(vid: str) -> dict:
        if vid not in svc.voyages:
            raise HTTPException(404, "unknown voyage")
        return svc.voyages[vid]

    @app.post("/voyages/{vid}/replan")
    def replan_voyage(vid: str, req: ReplanRequest):
        v = _voyage(vid)
        n = svc.check_scenarios(req.scenarios)
        world = svc.world(req.issued, n, v["request"].resolution_km, req.seed)
        try:
            position = world.grid.cell_of(req.lat, req.lon)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        r = svc.cfg.routing
        policy = ReplanPolicy(r.risk_budget, r.risk_estimator, r.confidence)
        try:
            decision = replan(world, position, v["candidate"].route, svc.vessel, policy, r.risk_weights,
                              r.connectivity, 0, data_age_hours=req.data_age_hours, seed=req.seed)
        except ValueError as exc:
            raise HTTPException(400, f"{exc}; send a position on the current route") from None
        if decision.action == "switch":
            v["candidate"] = decision.new_candidate
        else:
            v["candidate"] = Candidate(v["candidate"].labels, decision.previous_route, decision.previous_evaluation,
                                       decision.action == "keep", decision.previous_evaluation.p_breach_upper)
        v["world"] = world
        record = decision.record(issued=req.issued.isoformat(), event="replan")
        record.pop("previous", None)
        record.pop("new", None)
        v["events"].append(record)
        return {**record, "route": svc.candidate_payload(v["candidate"], world)}

    @app.get("/voyages/{vid}/history")
    def history(vid: str):
        return {"voyage_id": vid, "events": _voyage(vid)["events"]}

    @app.get("/voyages/{vid}/export")
    def export(vid: str, format: str = Query("geojson", pattern="^(geojson|csv)$")):  # noqa: A002
        v = _voyage(vid)
        gj = route_to_geojson(v["candidate"], v["world"], issued=v["world"].start.isoformat() + "Z")
        if format == "geojson":
            return gj
        buf = io.StringIO()
        rows = [f["properties"] for f in gj["features"][1:]]
        coords = [f["geometry"]["coordinates"] for f in gj["features"][1:]]
        w = csv.writer(buf)
        w.writerow(["waypoint_index", "lat", "lon", "planned_arrival_utc", "segment_breach_prob", "disclaimer"])
        for p, (lon, lat) in zip(rows, coords, strict=True):
            w.writerow([p["waypoint_index"], lat, lon, p["planned_arrival_utc"], p["segment_breach_prob"],
                        DISCLAIMER])
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f"attachment; filename=voyage_{vid}.csv"})

    if DASHBOARD_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def dashboard():
            return FileResponse(DASHBOARD_DIR / "index.html")

    return app


app = create_app() if os.environ.get("ANTROUTE_CONFIG") else None
