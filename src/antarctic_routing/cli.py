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


def cmd_fetch_sea_ice(args) -> int:
    from antarctic_routing.ingestion import sea_ice
    from antarctic_routing.ingestion.base import run_download

    cfg = load_config(args.config)
    d = cfg.domain
    root = Path(args.root)
    results = []
    for req in sea_ice.daily_requests(args.start, args.end, (d.lat_min, d.lat_max, d.lon_min, d.lon_max)):
        res = run_download(req, root, lambda r, dest: sea_ice.fetch_osisaf(r, dest))
        results.append({"date": req.start.isoformat(), "status": res.status,
                        "outputs": res.outputs, "warnings": res.warnings})
        print(f"  {req.start}  {res.status:8s} {'; '.join(res.warnings)[:100]}")
    write_json_artifact(root / "sea_ice" / "fetch-summary.json", {"product": sea_ice.PRODUCT, "results": results})
    ok = all(r["status"] == "passed" for r in results)
    print("All files available." if ok else "Some downloads did not complete - see fetch-summary.json.")
    return 0 if ok else 1


def cmd_build_dataset(args) -> int:
    import glob

    from antarctic_routing.ingestion.osisaf_reader import build_sea_ice_dataset

    t0, started = time.time(), utc_now()
    cfg = load_config(args.config)
    grid = PolarGrid.from_domain(cfg.domain, args.resolution_km or cfg.grid.resolution_km)
    paths = sorted({p for pattern in args.inputs for p in glob.glob(pattern, recursive=True)})
    if not paths:
        print("No input files matched.")
        return 1
    ds = build_sea_ice_dataset(paths, grid)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out)
    imputed = float(ds["imputed_mask"].mean())
    stage = StageResult(
        stage="harmonise_sea_ice", status="passed", execution_mode="real", software=SOFTWARE,
        inputs=[file_record(Path(p), "OSI-401-b") for p in paths],
        parameters={"resolution_km": grid.resolution_m / 1000, "crs": "EPSG:3031"},
        outputs=[file_record(out, "harmonised sea-ice dataset")],
        metrics={"days": int(ds.sizes["time"]), "imputed_fraction": imputed,
                 "land_fraction": float(ds["land_mask"].mean())},
        command=" ".join(["antroute", *args.argv]), started_at=started, finished_at=utc_now(),
        duration_seconds=round(time.time() - t0, 3),
    )
    write_json_artifact(out.with_suffix(".stage-result.json"), stage.to_dict())
    print(f"Wrote {out} ({ds.sizes['time']} days, {imputed:.1%} imputed cells)")
    return 0


def _forecast_data(args):
    """Load real harmonised data (--data) or generate a synthetic history; split seasons."""
    import xarray as xr

    from antarctic_routing.forecasting.dataset import issue_seasons
    from antarctic_routing.preprocessing.climatology import chronological_split
    from antarctic_routing.synthetic import synthetic_history

    cfg = load_config(args.config)
    if args.data:
        ds = xr.load_dataset(args.data)
    else:
        a, b = (int(v) for v in args.synthetic_seasons.split(":"))
        grid = PolarGrid.from_domain(cfg.domain, args.resolution_km or 25.0)
        ds = synthetic_history(cfg, grid, range(a, b), seed=args.seed)
    months = cfg.project.season_months
    seasons = sorted({s for s in issue_seasons(ds, range(ds.sizes["time"]), months) if s is not None})
    split = chronological_split(seasons, args.n_val, args.n_test)
    return cfg, ds, months, split


def cmd_train_forecast(args) -> int:
    from antarctic_routing.forecasting.train import TrainConfig, train_unet

    cfg, ds, months, split = _forecast_data(args)
    tc = TrainConfig(history_days=args.history_days or cfg.forecast.history_days, lead_days=args.lead_days,
                     epochs=args.epochs, base_channels=args.base_channels, seed=args.seed,
                     residual=not args.direct)
    print(f"Training on seasons {split['train']}, validating on {split['val']} "
          f"({ds.attrs.get('execution_mode', 'real')} data)")
    result = train_unet(ds, split["train"], split["val"], months, tc, args.out)
    for row in result.history:
        print(f"  epoch {row['epoch']:3d}  train {row['train_loss']:.4f}  val MAE {row['val_mae']:.4f}")
    print(f"Best epoch {result.best_epoch}; checkpoint {result.checkpoint}")
    return 0


def cmd_evaluate_forecast(args) -> int:
    from antarctic_routing.forecasting.evaluate import evaluate_forecasts
    from antarctic_routing.forecasting.train import load_model, unet_predictor
    from antarctic_routing.viz import plot_forecast_skill

    t0, started = time.time(), utc_now()
    cfg, ds, months, split = _forecast_data(args)
    model, meta = load_model(args.weights)
    hd, ld = meta["train_config"]["history_days"], meta["train_config"]["lead_days"]
    if set(meta["train_seasons"]) & set(split["test"]):
        raise SystemExit("refusing to evaluate: checkpoint was trained on a test season")
    report = evaluate_forecasts(ds, unet_predictor(model), split["train"], split["test"], hd, ld, months)
    out = Path(args.out)
    outputs = [write_json_artifact(out / "forecast_eval.json", report)]
    outputs.append(plot_forecast_skill(report, out / "forecast_skill.png", "Sea-ice U-Net vs baselines"))
    stage = StageResult(
        stage="forecast_evaluation", status="passed", execution_mode=report["execution_mode"], software=SOFTWARE,
        inputs=[file_record(Path(args.weights), "U-Net checkpoint")],
        parameters={"history_days": hd, "lead_days": ld, "train_seasons": split["train"],
                    "test_seasons": split["test"]},
        outputs=[file_record(p, "forecast_evaluation") for p in outputs],
        metrics={"mae": report["mae"], "skill_mae_vs": report["skill_mae_vs"]},
        command=" ".join(["antroute", *args.argv]), started_at=started, finished_at=utc_now(),
        duration_seconds=round(time.time() - t0, 3),
    )
    write_json_artifact(out / "stage-result.json", stage.to_dict())
    header = "lead  " + "  ".join(f"{m[:12]:>12s}" for m in report["methods"])
    print(header)
    for i, h in enumerate(report["leads"]):
        print(f"{h:4d}  " + "  ".join(f"{report['mae'][m][i]:12.4f}" for m in report["methods"]))
    return 0


def cmd_calibrate_forecast(args) -> int:
    from antarctic_routing.forecasting.calibration import evaluate_probabilities
    from antarctic_routing.forecasting.train import load_model, unet_predictor
    from antarctic_routing.viz import plot_reliability

    cfg, ds, months, split = _forecast_data(args)
    model, meta = load_model(args.weights)
    if set(meta["train_seasons"]) & (set(split["val"]) | set(split["test"])):
        raise SystemExit("refusing to calibrate: checkpoint was trained on a validation/test season")
    tc = meta["train_config"]
    report = evaluate_probabilities(
        ds, unet_predictor(model), split["train"], split["val"], split["test"], tc["history_days"],
        tc["lead_days"], months, tau=cfg.vessel.max_ice_concentration, n_members=args.members, seed=args.seed,
    )
    out = Path(args.out)
    write_json_artifact(out / "calibrator.json", {**report["calibrator"], "tau": report["tau"],
                                                  "weights_sha256": sha256_file(Path(args.weights))})
    write_json_artifact(out / "probability_eval.json", report)
    plot_reliability(report, out / "reliability.png", "Sea-ice hazard probability calibration")
    print("lead   Brier raw   calibrated   climatology")
    for i, h in enumerate(report["leads"]):
        print(f"{h:4d}  {report['brier']['raw'][i]:10.4f}  {report['brier']['calibrated'][i]:11.4f}"
              f"  {report['brier']['climatology'][i]:12.4f}")
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

    fs = sub.add_parser("fetch-sea-ice", help="download OSI SAF OSI-401-b daily files")
    fs.add_argument("--config", default=DEFAULT_CONFIG)
    fs.add_argument("--start", type=date.fromisoformat, required=True)
    fs.add_argument("--end", type=date.fromisoformat, required=True)
    fs.add_argument("--root", default="data/raw")
    fs.set_defaults(func=cmd_fetch_sea_ice)

    bd = sub.add_parser("build-dataset", help="harmonise OSI SAF files onto the EPSG:3031 grid")
    bd.add_argument("--config", default=DEFAULT_CONFIG)
    bd.add_argument("--inputs", nargs="+", required=True, help="file globs, e.g. 'data/raw/sea_ice/**/*.nc'")
    bd.add_argument("--out", default="data/processed/sea_ice.nc")
    bd.add_argument("--resolution-km", type=float, default=None)
    bd.set_defaults(func=cmd_build_dataset)

    def forecast_common(sp):
        sp.add_argument("--config", default=DEFAULT_CONFIG)
        src = sp.add_mutually_exclusive_group()
        src.add_argument("--data", default=None, help="harmonised NetCDF from build-dataset")
        src.add_argument("--synthetic-seasons", default="2004:2024", help="start:end (end exclusive)")
        sp.add_argument("--resolution-km", type=float, default=None)
        sp.add_argument("--history-days", type=int, default=None)
        sp.add_argument("--lead-days", type=int, default=7)
        sp.add_argument("--n-val", type=int, default=3)
        sp.add_argument("--n-test", type=int, default=3)
        sp.add_argument("--seed", type=int, default=42)

    tf = sub.add_parser("train-forecast", help="train the sea-ice U-Net")
    forecast_common(tf)
    tf.add_argument("--epochs", type=int, default=30)
    tf.add_argument("--base-channels", type=int, default=16)
    tf.add_argument("--direct", action="store_true", help="predict C directly instead of C_t + change")
    tf.add_argument("--out", default="models/unet")
    tf.set_defaults(func=cmd_train_forecast)

    ef = sub.add_parser("evaluate-forecast", help="score the U-Net against baselines per lead")
    forecast_common(ef)
    ef.add_argument("--weights", default="models/unet/best.pt")
    ef.add_argument("--out", default="reports/forecast")
    ef.set_defaults(func=cmd_evaluate_forecast)

    cf = sub.add_parser("calibrate-forecast", help="ensemble probabilities + isotonic calibration")
    forecast_common(cf)
    cf.add_argument("--weights", default="models/unet/best.pt")
    cf.add_argument("--members", type=int, default=20)
    cf.add_argument("--out", default="reports/calibration")
    cf.set_defaults(func=cmd_calibrate_forecast)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(argv)
    args.argv = argv
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
