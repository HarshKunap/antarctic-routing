import json
from pathlib import Path

from antarctic_routing.cli import main

CONFIG = str(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
FAST = ["--resolution-km", "25", "--scenarios", "80"]


def test_validate_config_prints_summary(capsys):
    assert main(["validate-config", "--config", CONFIG]) == 0
    out = capsys.readouterr().out
    assert "EPSG:3031" in out and "risk budget" in out.lower()


def test_demo_writes_all_artifacts(tmp_path):
    rc = main(["demo", "--config", CONFIG, "--departure", "2027-01-10", "--out", str(tmp_path), *FAST])
    assert rc == 0
    for name in ("plan.json", "route_map.png", "stage-result.json"):
        assert (tmp_path / name).is_file(), name
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["status"] in {"feasible", "infeasible"}
    assert plan["execution_mode"] == "controlled_synthetic"
    stage = json.loads((tmp_path / "stage-result.json").read_text())
    assert stage["execution_mode"] == "controlled_synthetic"
    assert stage["status"] in {"passed", "fallback"}
    if plan["status"] == "feasible":
        gj = json.loads((tmp_path / "recommended_route.geojson").read_text())
        assert gj["type"] == "FeatureCollection"
        assert (tmp_path / "recommended_route.csv").is_file()


def test_departures_sweep_writes_chart_and_table(tmp_path):
    rc = main(["departures", "--config", CONFIG, "--start", "2026-11-20", "--end", "2027-01-10",
               "--step-days", "17", "--out", str(tmp_path), *FAST])
    assert rc == 0
    sweep = json.loads((tmp_path / "departures.json").read_text())
    assert len(sweep["options"]) == 4
    assert (tmp_path / "departure_chart.png").is_file()


def test_build_dataset_from_osisaf_files(tmp_path):
    from datetime import date

    import xarray as xr

    from _osisaf_fixture import write_osisaf

    raw = tmp_path / "raw"
    raw.mkdir()
    for d in (1, 2):
        write_osisaf(raw / f"ice_{d}.nc", date(2024, 12, d))
    out = tmp_path / "sea_ice.nc"
    rc = main(["build-dataset", "--config", CONFIG, "--inputs", str(raw / "*.nc"), "--out", str(out),
               "--resolution-km", "25"])
    assert rc == 0
    with xr.open_dataset(out) as ds:
        assert ds.sizes["time"] == 2 and ds.attrs["execution_mode"] == "real"
    stage = json.loads(out.with_suffix(".stage-result.json").read_text())
    assert stage["status"] == "passed" and len(stage["inputs"]) == 2


def test_fetch_sea_ice_reports_blocked_or_failed_without_network(tmp_path, monkeypatch):
    import antarctic_routing.ingestion.sea_ice as sea_ice

    def offline(request, dest, timeout=60.0):
        raise ConnectionError("network unavailable")

    monkeypatch.setattr(sea_ice, "fetch_osisaf", offline)
    rc = main(["fetch-sea-ice", "--config", CONFIG, "--start", "2024-12-01", "--end", "2024-12-02",
               "--root", str(tmp_path)])
    assert rc == 1
    summary = json.loads((tmp_path / "sea_ice" / "fetch-summary.json").read_text())
    assert [r["status"] for r in summary["results"]] == ["failed", "failed"]


def test_train_and_evaluate_forecast_on_synthetic_history(tmp_path):
    common = ["--config", CONFIG, "--synthetic-seasons", "2010:2016", "--resolution-km", "50",
              "--history-days", "5", "--lead-days", "3", "--n-val", "1", "--n-test", "1"]
    rc = main(["train-forecast", *common, "--epochs", "2", "--base-channels", "8", "--out", str(tmp_path / "model")])
    assert rc == 0
    assert (tmp_path / "model" / "best.pt").is_file()
    rc = main(["evaluate-forecast", *common, "--weights", str(tmp_path / "model" / "best.pt"),
               "--out", str(tmp_path / "report")])
    assert rc == 0
    report = json.loads((tmp_path / "report" / "forecast_eval.json").read_text())
    assert report["test_seasons"] == [2015] and report["execution_mode"] == "controlled_synthetic"
    assert (tmp_path / "report" / "forecast_skill.png").is_file()
    stage = json.loads((tmp_path / "report" / "stage-result.json").read_text())
    assert stage["stage"] == "forecast_evaluation"


def test_calibrate_forecast_writes_report_calibrator_and_reliability_plot(tmp_path):
    common = ["--config", CONFIG, "--synthetic-seasons", "2010:2016", "--resolution-km", "50",
              "--history-days", "5", "--lead-days", "3", "--n-val", "1", "--n-test", "1"]
    assert main(["train-forecast", *common, "--epochs", "1", "--base-channels", "8",
                 "--out", str(tmp_path / "model")]) == 0
    rc = main(["calibrate-forecast", *common, "--weights", str(tmp_path / "model" / "best.pt"),
               "--members", "8", "--out", str(tmp_path / "cal")])
    assert rc == 0
    report = json.loads((tmp_path / "cal" / "probability_eval.json").read_text())
    assert report["val_seasons"] == [2014] and report["test_seasons"] == [2015]
    assert json.loads((tmp_path / "cal" / "calibrator.json").read_text())["leads"] == [1, 2, 3]
    assert (tmp_path / "cal" / "reliability.png").is_file()


def test_demo_with_iceberg_records_it_in_plan(tmp_path):
    rc = main(["demo", "--config", CONFIG, "--departure", "2027-01-10", "--out", str(tmp_path), *FAST,
               "--iceberg", "A23A:-60.0:-63.0"])
    assert rc == 0
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["icebergs"] == [{"id": "A23A", "lat": -60.0, "lon": -63.0}]
    assert "iceberg" in plan["data_description"].lower()


def test_demo_reads_latest_usnic_positions(tmp_path):
    csv_path = tmp_path / "usnic.csv"
    csv_path.write_text("Iceberg,Length (NM),Width (NM),Latitude,Longitude,Updated\n"
                        "A23A,38,32,60 0S,63 0W,12/01/2026\n"
                        "A23A,38,32,60 30S,62 0W,12/08/2026\n")
    rc = main(["demo", "--config", CONFIG, "--departure", "2027-01-10", "--out", str(tmp_path / "o"), *FAST,
               "--icebergs", str(csv_path)])
    assert rc == 0
    plan = json.loads((tmp_path / "o" / "plan.json").read_text())
    assert plan["icebergs"] == [{"id": "A23A", "lat": -60.5, "lon": -62.0}]


def test_trust_horizon_from_evaluation_report(tmp_path):
    seasons = {str(s): None for s in (2021, 2022, 2023, 2024, 2025)}
    report = {
        "methods": ["unet", "persistence", "climatology", "damped_persistence"], "leads": [1, 2, 3],
        "execution_mode": "controlled_synthetic",
        "by_season": {
            "unet": {s: [0.010, 0.015, 0.030] for s in seasons},
            "persistence": {s: [0.012, 0.020, 0.028] for s in seasons},
            "damped_persistence": {s: [0.011, 0.018, 0.027] for s in seasons},
            "climatology": {s: [0.030, 0.030, 0.030] for s in seasons},
        },
    }
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(report))
    assert main(["trust-horizon", "--report", str(path), "--out", str(tmp_path / "t")]) == 0
    th = json.loads((tmp_path / "t" / "trust_horizon.json").read_text())
    assert th["by_baseline"]["damped_persistence"]["trust_horizon_days"] == 2
    assert th["by_baseline"]["climatology"]["trust_horizon_days"] == 2
    assert th["trust_horizon_days"] == 2
    assert (tmp_path / "t" / "trust_horizon.png").is_file()
