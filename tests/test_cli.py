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
