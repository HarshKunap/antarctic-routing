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


def test_build_dataset_stage_records_product_family_and_actual_product(tmp_path):
    """Provenance keeps the family (OSI-401) apart from the product each file reports (OSI-401-b/-d)."""
    from datetime import date

    from _osisaf_fixture import write_osisaf

    raw = tmp_path / "raw"
    raw.mkdir()
    write_osisaf(raw / "ice_b.nc", date(2024, 12, 1))
    write_osisaf(raw / "ice_d.nc", date(2026, 9, 15), layout="osi401d")
    out = tmp_path / "sea_ice.nc"
    rc = main(["build-dataset", "--config", CONFIG, "--inputs", str(raw / "*.nc"), "--out", str(out),
               "--resolution-km", "25"])
    assert rc == 0
    stage = json.loads(out.with_suffix(".stage-result.json").read_text())
    by_name = {Path(r["path"]).name: r for r in stage["inputs"]}
    assert by_name["ice_b.nc"]["source"] == "OSI-401-b" and by_name["ice_b.nc"]["product_version"] == ""
    assert by_name["ice_d.nc"]["source"] == "OSI-401-d" and by_name["ice_d.nc"]["product_version"] == "4.1"
    assert {r["product_family"] for r in stage["inputs"]} == {"OSI-401"}
    assert stage["parameters"]["product_family"] == "OSI-401"
    assert stage["parameters"]["source_product"] == "OSI-401-b,OSI-401-d"


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


def test_plan_window_from_one_forecast_issue(tmp_path):
    common = ["--config", CONFIG, "--synthetic-seasons", "2010:2016", "--resolution-km", "50",
              "--history-days", "5", "--lead-days", "3", "--n-val", "1", "--n-test", "1"]
    assert main(["train-forecast", *common, "--epochs", "1", "--base-channels", "8",
                 "--out", str(tmp_path / "model")]) == 0
    trust = tmp_path / "trust.json"
    trust.write_text(json.dumps({"trust_horizon_days": 2}))
    rc = main(["plan-window", *common, "--weights", str(tmp_path / "model" / "best.pt"),
               "--issue", "2015-12-20", "--window-days", "4", "--members", "80",
               "--trust-report", str(trust), "--out", str(tmp_path / "w")])
    assert rc == 0
    out = json.loads((tmp_path / "w" / "plan_window.json").read_text())
    assert [o["lead_days"] for o in out["options"]] == [0, 1, 2, 3]
    assert out["issue"] == "2015-12-20" and out["trust_horizon_days"] == 2
    assert out["layer_source"][0] == "observed" and "climatology" in out["layer_source"]
    assert all(o["support"] in ("forecast-supported", "climatology-dominated") for o in out["options"])
    assert (tmp_path / "w" / "departure_window.png").is_file()


def test_replay_cli_writes_log_figure_and_summary(tmp_path):
    common = ["--config", CONFIG, "--synthetic-seasons", "2010:2016", "--resolution-km", "50",
              "--history-days", "5", "--lead-days", "3", "--n-val", "1", "--n-test", "1"]
    assert main(["train-forecast", *common, "--epochs", "1", "--base-channels", "8",
                 "--out", str(tmp_path / "model")]) == 0
    rc = main(["replay", *common, "--weights", str(tmp_path / "model" / "best.pt"), "--start", "2015-11-20",
               "--window-days", "4", "--members", "80", "--out", str(tmp_path / "r")])
    assert rc == 0
    summary = json.loads((tmp_path / "r" / "replay.json").read_text())
    assert summary["start"] == "2015-11-20" and "truth" in summary
    assert (tmp_path / "r" / "audit.jsonl").is_file()
    assert (tmp_path / "r" / "replay.png").is_file()


def test_backtest_cli_writes_summary_table_and_figure(tmp_path):
    common = ["--config", CONFIG, "--synthetic-seasons", "2010:2016", "--resolution-km", "50",
              "--history-days", "5", "--lead-days", "3", "--n-val", "1", "--n-test", "1"]
    assert main(["train-forecast", *common, "--epochs", "1", "--base-channels", "8",
                 "--out", str(tmp_path / "model")]) == 0
    rc = main(["backtest", *common, "--weights", str(tmp_path / "model" / "best.pt"),
               "--start-days", "11-18", "12-02", "--window-days", "4", "--members", "80",
               "--out", str(tmp_path / "b")])
    assert rc == 0
    out = json.loads((tmp_path / "b" / "backtest.json").read_text())
    assert out["starts"] == ["2015-11-18", "2015-12-02"]          # test season only
    assert set(out["summary"]) == {"planner", "naive", "ice_edge_buffer"}
    assert (tmp_path / "b" / "backtest.png").is_file()
    assert (tmp_path / "b" / "backtest_voyages.csv").is_file()


def test_sensitivity_cli_writes_grid_and_heatmap(tmp_path):
    rc = main(["sensitivity", "--config", CONFIG, "--departure", "2026-12-20", *FAST,
               "--lambdas", "0", "4", "--ks", "0.5", "0.7", "--out", str(tmp_path)])
    assert rc == 0
    out = json.loads((tmp_path / "sensitivity.json").read_text())
    assert len(out["grid"]) == 4 and out["baseline"]["lambda"] == 4.0
    assert (tmp_path / "sensitivity.png").is_file()


def test_voyage_brief_pdf(tmp_path):
    out = tmp_path / "brief.pdf"
    rc = main(["brief", "--config", CONFIG, "--departure", "2027-01-10", *FAST, "--out", str(out)])
    assert rc == 0
    data = out.read_bytes()
    assert data.startswith(b"%PDF")
    assert data.count(b"/Type /Page") - data.count(b"/Type /Pages") >= 2


def test_build_forcing_and_use_it_in_plan_window(tmp_path):
    from test_forcing import write_cmems, write_era5

    era5, cmems = write_era5(tmp_path / "era5.nc"), write_cmems(tmp_path / "cmems.nc")
    forcing = tmp_path / "forcing.nc"
    assert main(["build-forcing", "--config", CONFIG, "--era5", str(era5), "--cmems", str(cmems),
                 "--resolution-km", "50", "--out", str(forcing)]) == 0
    assert forcing.is_file() and forcing.with_suffix(".stage-result.json").is_file()
    common = ["--config", CONFIG, "--synthetic-seasons", "2010:2016", "--resolution-km", "50",
              "--history-days", "5", "--lead-days", "3", "--n-val", "1", "--n-test", "1"]
    assert main(["train-forecast", *common, "--epochs", "1", "--base-channels", "8",
                 "--out", str(tmp_path / "model")]) == 0
    rc = main(["plan-window", *common, "--weights", str(tmp_path / "model" / "best.pt"), "--forcing", str(forcing),
               "--issue", "2015-12-20", "--window-days", "2", "--members", "80",
               "--iceberg", "B1:-60.2:-62.6", "--out", str(tmp_path / "w")])
    assert rc == 0
    out = json.loads((tmp_path / "w" / "plan_window.json").read_text())
    assert out["forcing"] == str(forcing)
    assert out["icebergs"] == [{"id": "B1", "lat": -60.2, "lon": -62.6}]


def test_fetch_forcing_without_credentials_is_blocked(tmp_path, monkeypatch):
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    rc = main(["fetch-forcing", "--config", CONFIG, "--start", "2024-12-01", "--end", "2024-12-03",
               "--root", str(tmp_path / "raw")])
    assert rc == 1
    summary = json.loads((tmp_path / "raw" / "forcing-summary.json").read_text())
    assert {r["status"] for r in summary["results"]} <= {"blocked", "failed"}


def test_forecast_horizon_defaults_to_config_lead_days_end_to_end(tmp_path):
    """No --lead-days: config forecast.lead_days drives the samples, the model output and the evaluation."""
    import pytest
    import torch

    from antarctic_routing.config import load_config
    from antarctic_routing.forecasting.train import load_model

    lead = load_config(CONFIG).forecast.lead_days
    assert lead == 21
    common = ["--config", CONFIG, "--synthetic-seasons", "2010:2016", "--resolution-km", "50",
              "--history-days", "5", "--n-val", "1", "--n-test", "1"]
    assert main(["train-forecast", *common, "--epochs", "1", "--base-channels", "8",
                 "--out", str(tmp_path / "model")]) == 0
    model, meta = load_model(tmp_path / "model" / "best.pt")
    assert meta["train_config"]["lead_days"] == lead and meta["out_channels"] == lead
    with torch.no_grad():
        out = model(torch.zeros(1, meta["in_channels"], *meta["grid_shape"]))
    assert out.shape[1] == lead

    assert main(["evaluate-forecast", *common, "--weights", str(tmp_path / "model" / "best.pt"),
                 "--out", str(tmp_path / "report")]) == 0
    report = json.loads((tmp_path / "report" / "forecast_eval.json").read_text())
    assert report["leads"] == list(range(1, lead + 1))
    assert report["n_samples"] == 120 - 5 - lead + 1  # one test season, windows built with the 21-day horizon
    with pytest.raises(SystemExit, match="does not match the checkpoint"):
        main(["evaluate-forecast", *common, "--lead-days", "7", "--weights", str(tmp_path / "model" / "best.pt"),
              "--out", str(tmp_path / "report7")])


def test_evaluate_refuses_checkpoint_trained_on_another_resolution(tmp_path):
    import pytest

    common = ["--config", CONFIG, "--synthetic-seasons", "2010:2016", "--history-days", "5", "--lead-days", "3",
              "--n-val", "1", "--n-test", "1"]
    assert main(["train-forecast", *common, "--resolution-km", "50", "--epochs", "1", "--base-channels", "8",
                 "--out", str(tmp_path / "model")]) == 0
    with pytest.raises(SystemExit, match="resolution 100 km != trained 50 km"):
        main(["evaluate-forecast", *common, "--resolution-km", "100",
              "--weights", str(tmp_path / "model" / "best.pt"), "--out", str(tmp_path / "report")])
