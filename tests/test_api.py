"""HTTP API contract (Stage 11)."""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from antarctic_routing import DISCLAIMER
from antarctic_routing.api.main import create_app

CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
FAST = {"departure": "2026-12-20", "scenarios": 80, "resolution_km": 25, "scenario_routes": 0}
OPEN_WATER = {**FAST, "departure": "2027-01-10"}


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(CONFIG)) as c:
        yield c


def test_health_reports_version_and_disclaimer(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["disclaimer"] == DISCLAIMER and body["version"]


def test_config_endpoint_returns_validated_scenario(client):
    body = client.get("/config").json()
    assert body["config"]["routing"]["risk_budget"] == 0.05
    assert body["min_scenarios_for_budget"] == 73


def test_synchronous_route_plan_includes_map_and_candidates(client):
    r = client.post("/routes?wait=true", json=FAST)
    assert r.status_code == 200
    job = r.json()
    assert job["status"] == "done", job.get("error")
    res = job["result"]
    m = res["map"]
    assert len(m["p_ice"]) == m["nx"] * m["ny"] == len(m["land"])
    assert res["status"] in ("feasible", "infeasible")
    assert res["execution_mode"] == "controlled_synthetic"
    for cand in res["candidates"]:
        assert len(cand["xy_km"]) == len(cand["latlon"]) >= 2
        assert {"labels", "tags", "p_breach", "p_breach_upper", "expected_hours", "expected_fuel"} <= cand.keys()


def test_asynchronous_job_lifecycle(client):
    r = client.post("/routes", json=FAST)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    for _ in range(120):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.25)
    assert job["status"] == "done"


def test_unknown_job_is_404(client):
    assert client.get("/jobs/does-not-exist").status_code == 404


def test_too_few_scenarios_for_the_budget_is_rejected(client):
    r = client.post("/routes?wait=true", json={**FAST, "scenarios": 20})
    assert r.status_code == 422
    assert "73" in r.text


def test_route_request_with_iceberg(client):
    r = client.post("/routes?wait=true", json={**FAST, "icebergs": [{"id": "A23A", "lat": -60.2, "lon": -62.6}]})
    res = r.json()["result"]
    assert res["icebergs"][0]["id"] == "A23A"
    assert res["map"]["p_berg"] is not None


def test_departure_window_job(client):
    r = client.post("/departures?wait=true", json={"start": "2026-12-08", "end": "2026-12-20", "step_days": 6,
                                                   "scenarios": 80, "resolution_km": 25})
    res = r.json()["result"]
    assert len(res["options"]) == 3 and "rule" in res


def test_voyage_lifecycle_replan_history_export(client):
    v = client.post("/voyages", json=OPEN_WATER)
    assert v.status_code == 201, v.text
    voyage = v.json()
    vid = voyage["voyage_id"]
    assert voyage["route"]["latlon"]
    lat, lon = voyage["route"]["latlon"][3]
    r = client.post(f"/voyages/{vid}/replan", json={"lat": lat, "lon": lon, "issued": "2026-12-21",
                                                     "scenarios": 80, "data_age_hours": 6})
    assert r.status_code == 200, r.text
    assert r.json()["action"] in ("keep", "switch", "no_feasible_route")
    hist = client.get(f"/voyages/{vid}/history").json()
    assert [h["event"] for h in hist["events"]] == ["planned", "replan"]
    gj = client.get(f"/voyages/{vid}/export?format=geojson").json()
    assert gj["type"] == "FeatureCollection"
    assert gj["features"][0]["properties"]["disclaimer"] == DISCLAIMER
    csv = client.get(f"/voyages/{vid}/export?format=csv")
    assert csv.headers["content-type"].startswith("text/csv")
    assert "planned_arrival_utc" in csv.text.splitlines()[0]


def test_replan_position_off_route_is_a_clear_400(client):
    vid = client.post("/voyages", json=OPEN_WATER).json()["voyage_id"]
    r = client.post(f"/voyages/{vid}/replan", json={"lat": -56.0, "lon": -70.5, "issued": "2026-12-21",
                                                     "scenarios": 80})
    assert r.status_code == 400 and "route" in r.json()["detail"]


def test_unknown_voyage_is_404(client):
    assert client.get("/voyages/nope/history").status_code == 404


def test_voyage_is_refused_when_no_route_meets_the_budget(client):
    r = client.post("/voyages", json={**FAST, "departure": "2026-11-20"})
    assert r.status_code == 409
    assert "risk budget" in r.json()["detail"]
