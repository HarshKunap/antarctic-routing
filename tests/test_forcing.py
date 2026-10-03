"""Real forcing: ERA5 10 m wind and CMEMS surface currents onto the polar grid."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from antarctic_routing.config import DomainSection
from antarctic_routing.ingestion.forcing import build_forcing, load_forcing
from antarctic_routing.preprocessing.grid import PolarGrid

DOMAIN = DomainSection(lat_min=-66.0, lat_max=-55.0, lon_min=-72.0, lon_max=-52.0)
LAT = np.arange(-45.0, -75.25, -0.25)   # ERA5 style: descending latitude
LON = np.arange(-85.0, -34.75, 0.25)


def write_era5(path: Path, u=10.0, v=0.0, time_name="valid_time"):
    t = [datetime(2024, 12, 1) + timedelta(hours=6 * k) for k in range(8)]
    shape = (len(t), LAT.size, LON.size)
    xr.Dataset({"u10": ((time_name, "latitude", "longitude"), np.full(shape, u, np.float32)),
                "v10": ((time_name, "latitude", "longitude"), np.full(shape, v, np.float32))},
               coords={time_name: t, "latitude": LAT, "longitude": LON}).to_netcdf(path)
    return path


def write_cmems(path: Path, uo=0.3, vo=0.0):
    t = [datetime(2024, 12, 1) + timedelta(days=k) for k in range(3)]
    lat = LAT[::-1]                         # CMEMS style: ascending latitude, with a depth axis
    shape = (len(t), 2, lat.size, LON.size)
    data = np.full(shape, uo, np.float32)
    data[:, 1] = 99.0                       # deeper level must be ignored
    xr.Dataset({"uo": (("time", "depth", "latitude", "longitude"), data),
                "vo": (("time", "depth", "latitude", "longitude"), np.full(shape, vo, np.float32))},
               coords={"time": t, "depth": [0.49, 1.54], "latitude": lat, "longitude": LON}).to_netcdf(path)
    return path


@pytest.fixture(scope="module")
def grid():
    return PolarGrid.from_domain(DOMAIN, resolution_km=25)


def test_wind_and_current_are_regridded_and_rotated(tmp_path, grid):
    ds = build_forcing(grid, era5_path=write_era5(tmp_path / "e.nc"), cmems_path=write_cmems(tmp_path / "c.nc"))
    lam = np.deg2rad(grid.lon2d)
    assert np.allclose(ds["wind_x"].values, 10.0 * np.cos(lam), atol=1e-4)
    assert np.allclose(ds["wind_y"].values, -10.0 * np.sin(lam), atol=1e-4)
    assert np.allclose(np.hypot(ds["current_x"], ds["current_y"]), 0.3, atol=1e-5)  # surface level only
    assert ds.attrs["execution_mode"] == "real" and ds.attrs["crs"] == "EPSG:3031"
    assert len(ds.attrs["source_sha256"].split(",")) == 2


def test_old_era5_time_name_is_supported(tmp_path, grid):
    ds = build_forcing(grid, era5_path=write_era5(tmp_path / "e.nc", u=0.0, v=5.0, time_name="time"))
    assert np.allclose(np.hypot(ds["wind_x"], ds["wind_y"]), 5.0, atol=1e-4)
    assert "current_x" not in ds


def test_missing_coverage_is_zero_filled_and_reported(tmp_path, grid):
    path = tmp_path / "c.nc"
    write_cmems(path)
    with xr.open_dataset(path) as src:
        small = src.sel(latitude=slice(-62, -55)).load()   # does not cover the whole grid
    small.to_netcdf(tmp_path / "small.nc")
    ds = build_forcing(grid, cmems_path=tmp_path / "small.nc")
    assert np.isfinite(ds["current_x"].values).all()
    assert 0 < ds.attrs["current_filled_fraction"] < 1


def test_load_forcing_returns_tuples_matching_grid(tmp_path, grid):
    out = tmp_path / "forcing.nc"
    ds = build_forcing(grid, era5_path=write_era5(tmp_path / "e.nc"), cmems_path=write_cmems(tmp_path / "c.nc"))
    ds.to_netcdf(out)
    currents, winds = load_forcing(out, expected_shape=grid.shape)
    assert currents[0].shape == grid.shape and winds[1].shape == grid.shape
    with pytest.raises(ValueError, match="grid"):
        load_forcing(out, expected_shape=(3, 3))


def test_build_forcing_requires_at_least_one_source(grid):
    with pytest.raises(ValueError):
        build_forcing(grid)
