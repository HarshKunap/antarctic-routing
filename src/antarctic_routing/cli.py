"""Command-line entry point: ``antroute`` (or ``python -m antarctic_routing.cli``).

Commands
--------
validate-config   Validate config.yaml and print the scenario summary.
demo              Plan risk-budgeted routes for one departure (controlled-synthetic data).
departures        Sweep a departure window and select a date under the risk budget.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from pyproj import Geod

from antarctic_routing import DISCLAIMER, __version__
from antarctic_routing.common.provenance import (
    StageResult,
    file_record,
    sha256_file,
    utc_now,
    write_json_artifact,
)
from antarctic_routing.config import ProjectConfig, load_config, min_scenarios_for_budget
from antarctic_routing.export import route_to_csv, route_to_geojson
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.routing.candidates import plan_candidates
from antarctic_routing.routing.departure import sweep_departures
from antarctic_routing.routing.fuel import VesselModel
from antarctic_routing.synthetic import generate_synthetic
from antarctic_routing.viz import plot_departures, plot_plan

DEFAULT_CONFIG = "config/config.yaml"
SOFTWARE = {"name": "antarctic-routing", "version": __version__}


def _horizon_days(cfg: ProjectConfig) -> int:
    o, d = cfg.route.origin, cfg.route.destination
    km = Geod(ellps="WGS84").inv(o.lon, o.lat, d.lon, d.lat)[2] / 1000.0
    hours = 1.5 * km / (0.5 * cfg.vessel.cruise_speed_kmh)  # generous: detours + ice slowdown
    return int(min(cfg.forecast.lead_days, math.ceil(hours / cfg.grid.time_step_hours) + 1))


def _setup(args):
    cfg = load_config(args.config)
    grid = PolarGrid.from_domain(cfg.domain, args.resolution_km or cfg.grid.resolution_km)
    n_scen = args.scenarios or cfg.forecast.route_scenarios
    o = grid.cell_of(cfg.route.origin.lat, cfg.route.origin.lon)
    d = grid.cell_of(cfg.route.destination.lat, cfg.route.destination.lon)
    return cfg, grid, n_scen, o, d


def _plan_dict(plan, world) -> dict:
    return {
        "status": plan.status,
        "explanation": plan.explanation,
        "risk_budget": plan.risk_budget,
        "estimator": plan.estimator,
        "execution_mode": world.execution_mode,
        "data_description": world.description,
        "departure_utc": world.start.isoformat() + "Z",
        "n_scenarios": world.n_scenarios,
        "recommended": plan.recommended.labels if plan.recommended else None,
        "candidates": [
            {"labels": c.labels, "tags": c.tags, "feasible": c.feasible,
             "risk_statistic": c.risk_statistic, **c.evaluation.summary()}
            for c in plan.candidates
        ],
        "warnings": plan.warnings,
        "disclaimer": DISCLAIMER,
    }


def cmd_validate(args) -> int:
    cfg = load_config(args.config)
    v, r = cfg.vessel, cfg.routing
    print(f"Config OK: {args.config}")
    print(f"  Region      : {cfg.project.region} ({cfg.domain.crs}), season months {cfg.project.season_months}")
    print(f"  Route       : ({cfg.route.origin.lat}, {cfg.route.origin.lon}) -> "
          f"({cfg.route.destination.lat}, {cfg.route.destination.lon})")
    print(f"  Departures  : {cfg.route.departure_start} .. {cfg.route.departure_end}")
    print(f"  Vessel      : {v.name}, {v.cruise_speed_knots} kn ({v.cruise_speed_kmh:.1f} km/h), "
          f"ice limit tau_v={v.max_ice_concentration}")
    print(f"  Risk budget : {r.risk_budget:.1%} via {r.risk_estimator} at {r.confidence:.0%} confidence; "
          f"needs >= {min_scenarios_for_budget(r.risk_budget, r.confidence)} scenarios "
          f"(configured {cfg.forecast.route_scenarios})")
    if "REPLACE" in v.ice_class:
        print("  WARNING     : vessel.ice_class is a placeholder - replace with the verified class.")
    return 0


def cmd_demo(args) -> int:
    t0 = time.time()
    started = utc_now()
    cfg, grid, n_scen, o, d = _setup(args)
    dep = args.departure or cfg.route.departure_start
    world = generate_synthetic(cfg, grid, dep, _horizon_days(cfg), n_scen, seed=args.seed)
    vessel = VesselModel.from_config(cfg)
    plan = plan_candidates(
        world, vessel, o, d, cfg.routing.risk_budget, cfg.routing.risk_weights,
        cfg.routing.risk_estimator, cfg.routing.connectivity, args.scenario_routes,
        cfg.routing.confidence, args.seed,
    )
    out = Path(args.out)
    outputs = [write_json_artifact(out / "plan.json", _plan_dict(plan, world))]
    outputs.append(plot_plan(world, plan, vessel.tau, out / "route_map.png",
                             f"Antarctic route plan - departure {dep.isoformat()} ({n_scen} joint scenarios)"))
    cfg_sha = sha256_file(Path(args.config))
    if plan.recommended:
        gj = route_to_geojson(plan.recommended, world, issued=world.start.isoformat() + "Z", config_sha256=cfg_sha)
        outputs.append(write_json_artifact(out / "recommended_route.geojson", gj))
        outputs.append(route_to_csv(plan.recommended, world, out / "recommended_route.csv"))

    rec = plan.recommended.evaluation.summary() if plan.recommended else {}
    stage = StageResult(
        stage="route_planning", status="passed", execution_mode=world.execution_mode, software=SOFTWARE,
        inputs=[file_record(Path(args.config), "scenario configuration")],
        parameters={"departure": dep.isoformat(), "n_scenarios": n_scen, "seed": args.seed,
                    "resolution_km": grid.resolution_m / 1000, "risk_budget": cfg.routing.risk_budget,
                    "estimator": cfg.routing.risk_estimator},
        outputs=[file_record(p, "route_planning") for p in outputs],
        metrics={"plan_status": plan.status, "n_candidates": len(plan.candidates), **rec},
        command=" ".join(["antroute", *args.argv]),
        started_at=started, finished_at=utc_now(), duration_seconds=round(time.time() - t0, 3),
        warnings=plan.warnings + ["controlled_synthetic data: not a real forecast"],
    )
    write_json_artifact(out / "stage-result.json", stage.to_dict())
    print(plan.explanation)
    print(f"Artifacts written to {out}/")
    return 0


def cmd_departures(args) -> int:
    cfg, grid, n_scen, o, d = _setup(args)
    start = args.start or cfg.route.departure_start
    end = args.end or cfg.route.departure_end
    dates, cur = [], start
    while cur <= end:
        dates.append(cur)
        cur += timedelta(days=args.step_days)
    days = _horizon_days(cfg)
    sweep = sweep_departures(
        dates, lambda dd: generate_synthetic(cfg, grid, dd, days, n_scen, seed=args.seed),
        VesselModel.from_config(cfg), o, d, cfg.routing.risk_budget, cfg.routing.risk_weights,
        cfg.routing.risk_estimator, cfg.routing.connectivity, args.scenario_routes,
        cfg.routing.confidence, args.seed,
    )
    out = Path(args.out)
    payload = {**sweep.to_dict(), "execution_mode": "controlled_synthetic", "n_scenarios": n_scen,
               "disclaimer": DISCLAIMER}
    write_json_artifact(out / "departures.json", payload)
    plot_departures(sweep, cfg.routing.risk_budget, out / "departure_chart.png",
                    f"Departure window {start} .. {end} (controlled-synthetic, {n_scen} scenarios/date)")
    for opt in sweep.options:
        flag = "OK " if opt.feasible else "-- "
        print(f"  {flag}{opt.departure}  P(breach) UB {opt.p_breach_upper:6.1%}  "
              f"E[t] {opt.expected_hours:6.1f} h  E[fuel] {opt.expected_fuel:7.0f}")
    print(sweep.explanation)
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="antroute", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"antarctic-routing {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, out_default):
        sp.add_argument("--config", default=DEFAULT_CONFIG)
        sp.add_argument("--out", default=out_default)
        sp.add_argument("--scenarios", type=int, default=None, help="override forecast.route_scenarios")
        sp.add_argument("--resolution-km", type=float, default=None, help="override grid.resolution_km")
        sp.add_argument("--scenario-routes", type=int, default=3, help="per-scenario optimal candidates")
        sp.add_argument("--seed", type=int, default=42)

    v = sub.add_parser("validate-config", help="validate configuration")
    v.add_argument("--config", default=DEFAULT_CONFIG)
    v.set_defaults(func=cmd_validate)

    dm = sub.add_parser("demo", help="plan routes for one departure")
    common(dm, "artifacts/demo")
    dm.add_argument("--departure", type=date.fromisoformat, default=None)
    dm.set_defaults(func=cmd_demo)

    dp = sub.add_parser("departures", help="sweep a departure window")
    common(dp, "artifacts/departures")
    dp.add_argument("--start", type=date.fromisoformat, default=None)
    dp.add_argument("--end", type=date.fromisoformat, default=None)
    dp.add_argument("--step-days", type=int, default=1)
    dp.set_defaults(func=cmd_departures)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(argv)
    args.argv = argv
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
