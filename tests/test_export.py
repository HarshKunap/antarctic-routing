import csv
import json

from _worlds import A, B, _vessel, _world
from antarctic_routing import DISCLAIMER
from antarctic_routing.export import route_to_csv, route_to_geojson
from antarctic_routing.routing.candidates import plan_candidates


def _plan():
    world = _world(k=80)
    result = plan_candidates(world, _vessel(), A, B, risk_budget=0.05, risk_weights=[0, 100],
                             n_scenario_routes=0)
    return world, result


def test_geojson_uses_lon_lat_order_and_carries_disclaimer(tmp_path):
    world, result = _plan()
    rec = result.recommended
    gj = route_to_geojson(rec, world, issued="2026-12-01T00:00:00Z", config_sha256="abc")
    line = gj["features"][0]
    assert line["geometry"]["type"] == "LineString"
    lon0, lat0 = line["geometry"]["coordinates"][0]
    r, c = rec.route.cells[0]
    assert lat0 == round(float(world.grid.lat2d[r, c]), 6)
    assert lon0 == round(float(world.grid.lon2d[r, c]), 6)
    props = line["properties"]
    assert props["disclaimer"] == DISCLAIMER
    assert props["execution_mode"] == "controlled_synthetic"
    assert props["config_sha256"] == "abc"
    assert 0 <= props["p_breach"] <= 1
    json.dumps(gj)


def test_geojson_waypoints_have_planned_arrival_and_segment_risk():
    world, result = _plan()
    gj = route_to_geojson(result.recommended, world, issued="2026-12-01T00:00:00Z")
    points = [f for f in gj["features"] if f["geometry"]["type"] == "Point"]
    assert len(points) == len(result.recommended.route.cells)
    assert points[0]["properties"]["planned_arrival_utc"] == "2026-12-01T00:00:00Z"
    assert {"segment_breach_prob", "waypoint_index"} <= points[1]["properties"].keys()


def test_csv_export_rows_match_waypoints(tmp_path):
    world, result = _plan()
    path = route_to_csv(result.recommended, world, tmp_path / "route.csv")
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == len(result.recommended.route.cells)
    assert {"lat", "lon", "planned_arrival_utc", "segment_breach_prob", "disclaimer"} <= rows[0].keys()
    assert rows[0]["disclaimer"] == DISCLAIMER
