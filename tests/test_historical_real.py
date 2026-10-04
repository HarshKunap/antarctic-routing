"""Real Historical Data on the real archive: the API reproduces the frozen 2023-11-14 run exactly.

Runs only where the real-data archive (``ANTROUTE_DATA_ROOT``, default /mnt/project-files/real-data) and
PyTorch are present; the inputs are kept outside Git.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from antarctic_routing.api.main import create_app

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "config.yaml"
REAL_DATA = Path(os.environ.get("ANTROUTE_DATA_ROOT", "/mnt/project-files/real-data"))
SPEC = json.loads((REPO / "config" / "real_historical.json").read_text())
FROZEN = json.loads((REPO / "docs" / "frozen_demo" / "frozen_demo_2023-11-14.json").read_text())
FROZEN_WINDOW = json.loads((REPO / "deploy" / "bundle" / "plan_window.json").read_text())

pytestmark = pytest.mark.skipif(
    not (REAL_DATA / SPEC["inputs"]["sea_ice"]["path"]).is_file() or importlib.util.find_spec("torch") is None,
    reason="the real-data archive and PyTorch are kept outside Git")


@pytest.fixture(scope="module")
def client():
    old = os.environ.get("ANTROUTE_DATA_ROOT")
    os.environ["ANTROUTE_DATA_ROOT"] = str(REAL_DATA)
    try:
        with TestClient(create_app(CONFIG)) as c:
            yield c
    finally:
        if old is None:
            os.environ.pop("ANTROUTE_DATA_ROOT", None)
        else:
            os.environ["ANTROUTE_DATA_ROOT"] = old


def test_status_and_dates_cover_six_seasons(client):
    st = client.get("/real/historical/status").json()
    assert st["status"] == "available" and st["label"] == "Real Historical Data"
    assert "reanalysis" in st["hindsight_forcing"]
    d = client.get("/real/historical/dates").json()
    assert (len(d["route_dates"]), len(d["window_dates"])) == (596, 521)
    seasons = d["seasons"]["window"]
    assert [s["season"] for s in seasons] == ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
    assert [s["out_of_sample"] for s in seasons] == [False, False, False, False, True, True]


def test_route_reproduces_the_frozen_selected_route(client):
    r = client.post("/real/historical/routes?wait=true", json={"issue": "2023-11-14"}).json()
    assert r["status"] == "done", r.get("error")
    res = r["result"]
    rec = res["candidates"][res["recommended_index"]]
    exp = FROZEN["expected"]
    assert (rec["expected_hours"], rec["expected_fuel"], rec["distance_km"], rec["p_breach_upper"]) == \
        (exp["expected_hours"], exp["expected_fuel"], exp["distance_km"], exp["p_breach_upper"])
    assert rec["breaches"] == exp["breaches"] and len(rec["xy_km"]) == exp["route_cells"]
    assert (res["execution_mode"], res["data_status"], res["data_label"]) == ("real", "historical",
                                                                              "Real Historical Data")
    assert res["icebergs"]["list_date"] == "2023-11-09" and len(res["icebergs"]["drifted"]) == 7
    assert res["historical"]["season"]["out_of_sample"] is True
    assert res["historical"]["probability_calibration"]["applied_in_route_risk"] is False


def test_departure_window_matches_the_frozen_plan_window(client):
    r = client.post("/real/historical/departures?wait=true", json={"issue": "2023-11-14"}).json()
    assert r["status"] == "done", r.get("error")
    res = r["result"]
    keys = ("departure", "feasible", "expected_hours", "expected_fuel", "p_breach", "p_breach_upper", "lead_days",
            "forecast_fraction", "support")
    assert [{k: o[k] for k in keys} for o in res["options"]] == \
        [{k: o[k] for k in keys} for o in FROZEN_WINDOW["options"]]
    assert res["selected"] == FROZEN_WINDOW["selected"] and res["layer_source"] == FROZEN_WINDOW["layer_source"]


def test_out_of_coverage_dates_are_refused_with_the_reason(client):
    r = client.post("/real/historical/routes?wait=true", json={"issue": "2022-11-15"})
    assert r.status_code == 422 and "14 contiguous" in r.json()["detail"]["reason"]
    r = client.post("/real/historical/departures?wait=true", json={"issue": "2024-02-20"})
    assert r.status_code == 422 and "past the end of the season" in r.json()["detail"]["reason"]


def test_voyage_replan_and_replay_with_icebergs(client):
    v = client.post("/real/historical/voyages", json={"issue": "2023-11-14"})
    assert v.status_code == 201
    vid, route = v.json()["voyage_id"], v.json()["route"]["latlon"]
    wp = route[len(route) // 3]
    r = client.post(f"/real/historical/voyages/{vid}/replan", json={"lat": wp[0], "lon": wp[1], "issued": "2023-11-15"})
    assert r.status_code == 200 and r.json()["action"] in ("keep", "switch", "no_feasible_route")
    assert r.json()["usnic_list"] == "2023-11-09" and r.json()["data_label"] == "Real Historical Data"
    events = client.get(f"/voyages/{vid}/history").json()["events"]
    assert [e.get("event") for e in events] == ["planned", "replan"]
    assert client.post(f"/voyages/{vid}/replan", json={"lat": wp[0], "lon": wp[1],
                                                       "issued": "2023-11-15"}).status_code == 409
    props = client.get(f"/voyages/{vid}/export").json()["features"][0]["properties"]
    assert props["data_label"] == "Real Historical Data" and "reanalysis" in props["hindsight_forcing"]
    header = client.get(f"/voyages/{vid}/export?format=csv").text.splitlines()[0]
    assert header.endswith(",disclaimer,data_label,hindsight_forcing")
    rp = client.post("/real/historical/replay?wait=true", json={"start": "2023-11-14", "max_wait_days": 0}).json()
    assert rp["status"] == "done", rp.get("error")
    res = rp["result"]
    assert res["iceberg_hazard"] is True and res["departure"] == "2023-11-14" and res["arrived"] is True
    assert {"berg_footprint_cells", "berg_min_distance_km"} <= res["truth"]["planner"].keys()
    assert res["map"]["observed_date"] == "2023-11-14" and len(res["observed_bergs"]) == 7
