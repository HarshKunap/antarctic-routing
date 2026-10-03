import json
from datetime import date
from pathlib import Path

import pytest

from antarctic_routing.config import DomainSection
from antarctic_routing.ingestion.base import (
    DownloadRequest,
    MissingCredentials,
    MissingDependency,
    run_download,
)
from antarctic_routing.ingestion.cmems import cmems_subset_kwargs
from antarctic_routing.ingestion.era5 import era5_request
from antarctic_routing.ingestion.icebergs import parse_coordinate, read_iceberg_positions
from antarctic_routing.ingestion.sea_ice import osisaf_url

DOMAIN = DomainSection(lat_min=-66.0, lat_max=-55.0, lon_min=-72.0, lon_max=-52.0)


def _req(**kw):
    base = dict(source="era5", product="reanalysis-era5-single-levels", variables=("u10", "v10"),
                start=date(2024, 12, 1), end=date(2024, 12, 3), bbox=(-66.0, -55.0, -72.0, -52.0))
    base.update(kw)
    return DownloadRequest(**base)


def test_request_key_is_deterministic_and_parameter_sensitive():
    assert _req().key() == _req().key()
    assert _req().key() != _req(end=date(2024, 12, 4)).key()
    assert _req().key() != _req(variables=("u10",)).key()


def test_successful_download_writes_file_manifest_and_checksum(tmp_path):
    def fetch(request, dest):
        dest.write_bytes(b"netcdf-bytes")
        return {"source_url": "https://example.invalid/x.nc", "product_version": "v1"}

    result = run_download(_req(), tmp_path, fetch, suffix=".nc")
    assert result.status == "passed" and result.execution_mode == "real"
    out = result.outputs[0]
    assert out["sha256"] and Path(out["path"]).is_file()
    manifest = json.loads(next(tmp_path.rglob("*.manifest.json")).read_text())
    assert manifest["request"]["variables"] == ["u10", "v10"]
    assert manifest["product_version"] == "v1"
    assert manifest["sha256"] == out["sha256"]


def test_second_run_uses_cache_without_fetching(tmp_path):
    calls = []

    def fetch(request, dest):
        calls.append(1)
        dest.write_bytes(b"abc")
        return {}

    run_download(_req(), tmp_path, fetch)
    second = run_download(_req(), tmp_path, fetch)
    assert len(calls) == 1
    assert any("cached" in w for w in second.warnings)


def test_missing_credentials_is_blocked_not_faked(tmp_path):
    def fetch(request, dest):
        raise MissingCredentials("CDS API key not configured")

    result = run_download(_req(), tmp_path, fetch)
    assert result.status == "blocked"
    assert not list(tmp_path.rglob("*.nc"))


def test_missing_dependency_is_blocked(tmp_path):
    def fetch(request, dest):
        raise MissingDependency("cdsapi is not installed")

    assert run_download(_req(), tmp_path, fetch).status == "blocked"


def test_failed_download_leaves_no_partial_file(tmp_path):
    def fetch(request, dest):
        dest.write_bytes(b"half")
        raise ConnectionError("reset")

    result = run_download(_req(), tmp_path, fetch)
    assert result.status == "failed"
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


def test_osisaf_url_pattern():
    url = osisaf_url(date(2024, 12, 5))
    assert url.startswith("https://thredds.met.no/thredds/fileServer/osisaf/met.no/ice/conc/2024/12/")
    assert url.endswith("ice_conc_sh_polstere-100_multi_202412051200.nc")


def test_era5_request_uses_north_west_south_east_area():
    req = era5_request(DOMAIN, date(2024, 12, 30), date(2025, 1, 2))
    assert req["area"] == [-55.0, -72.0, -66.0, -52.0]
    assert "10m_u_component_of_wind" in req["variable"]
    assert set(req["year"]) == {"2024", "2025"}
    assert req["data_format"] == "netcdf"


def test_cmems_subset_kwargs():
    kw = cmems_subset_kwargs(DOMAIN, date(2024, 12, 1), date(2024, 12, 3), "out.nc")
    assert kw["minimum_latitude"] == -66.0 and kw["maximum_longitude"] == -52.0
    assert set(kw["variables"]) == {"uo", "vo", "thetao"}
    assert kw["start_datetime"].startswith("2024-12-01")


@pytest.mark.parametrize(
    "text,expected",
    [("-65.5", -65.5), ("65.5S", -65.5), ("65 30S", -65.5), ("65 30'S", -65.5), ("58 15W", -58.25), ("12.0E", 12.0)],
)
def test_parse_coordinate_formats(text, expected):
    assert parse_coordinate(text) == pytest.approx(expected)


def test_read_iceberg_positions_normalises_columns(tmp_path):
    path = tmp_path / "usnic.csv"
    path.write_text(
        "Iceberg,Length (NM),Width (NM),Latitude,Longitude,Updated\n"
        "A23A,38,32,61 30S,46 0W,12/01/2024\n"
        "D30B,10,5,-63.25,-57.5,12/02/2024\n"
    )
    rows = read_iceberg_positions(path)
    assert rows[0]["iceberg_id"] == "A23A"
    assert rows[0]["lat"] == pytest.approx(-61.5) and rows[0]["lon"] == pytest.approx(-46.0)
    assert rows[0]["length_km"] == pytest.approx(38 * 1.852)
    assert rows[1]["date"] == date(2024, 12, 2)
