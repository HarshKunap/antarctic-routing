"""OSI SAF OSI-401-b reader tests on fixture files that follow the product layout.

The fixtures mimic the documented structure (km coordinates, own polar
stereographic CRS with lat_ts=-70 on the Hughes ellipsoid, percent concentration,
CF flag-coded status). They are not real observations.
"""

from datetime import date

import numpy as np
import pytest

from _osisaf_fixture import _fixture_conc, write_osisaf
from antarctic_routing.config import DomainSection
from antarctic_routing.ingestion.osisaf_reader import build_sea_ice_dataset, read_osisaf
from antarctic_routing.preprocessing.grid import PolarGrid

DOMAIN = DomainSection(lat_min=-66.0, lat_max=-55.0, lon_min=-72.0, lon_max=-52.0)


@pytest.fixture(scope="module")
def grid():
    return PolarGrid.from_domain(DOMAIN, resolution_km=25)


def test_reader_converts_percent_km_and_reads_crs(tmp_path):
    src = read_osisaf(write_osisaf(tmp_path / "a.nc", date(2024, 12, 1)))
    assert src.date == date(2024, 12, 1)
    assert np.nanmax(src.conc) <= 1.0 and np.nanmin(src.conc) >= 0.0
    assert np.all(np.diff(src.x) > 0) and np.all(np.diff(src.y) > 0)  # sorted ascending, metres
    assert np.abs(src.x).max() > 1e5  # metres, not km
    cf = src.crs.to_cf()
    assert cf["standard_parallel"] == -70.0 and cf["semi_major_axis"] == 6378273.0


def test_regridded_field_matches_analytic_latitude_profile(tmp_path, grid):
    ds = build_sea_ice_dataset([write_osisaf(tmp_path / "a.nc", date(2024, 12, 1))], grid)
    got = ds["ice_concentration"].isel(time=0).values
    expected = _fixture_conc(grid.lat2d) / 100.0
    ocean = ~ds["land_mask"].values
    assert np.nanmax(np.abs(got[ocean] - expected[ocean])) < 0.03
    assert ds.attrs["crs"] == "EPSG:3031"
    assert ds.attrs["execution_mode"] == "real"


def test_land_flags_become_land_mask_and_missing_is_imputed_and_flagged(tmp_path, grid):
    path = write_osisaf(tmp_path / "a.nc", date(2024, 12, 1),
                        land_box=(-61.0, -60.0, -64.0, -62.0), missing_box=(-58.5, -58.2, -57.0, -56.4))
    ds = build_sea_ice_dataset([path], grid)
    land = ds["land_mask"].values
    imputed = ds["imputed_mask"].isel(time=0).values
    conc = ds["ice_concentration"].isel(time=0).values
    assert land.any() and not (land & imputed).any()
    assert imputed.any()
    assert np.isfinite(conc[~land]).all()
    assert np.isnan(conc[land]).all()


def test_daily_files_are_stacked_in_date_order_with_provenance(tmp_path, grid):
    paths = [write_osisaf(tmp_path / f"{d}.nc", date(2024, 12, d)) for d in (3, 1, 2)]
    ds = build_sea_ice_dataset(paths, grid)
    days = [str(t)[:10] for t in ds["time"].values]
    assert days == ["2024-12-01", "2024-12-02", "2024-12-03"]
    assert len(ds.attrs["source_sha256"].split(",")) == 3


def test_duplicate_dates_are_rejected(tmp_path, grid):
    a = write_osisaf(tmp_path / "a.nc", date(2024, 12, 1))
    b = write_osisaf(tmp_path / "b.nc", date(2024, 12, 1))
    with pytest.raises(ValueError, match="duplicate"):
        build_sea_ice_dataset([a, b], grid)
