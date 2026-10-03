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
