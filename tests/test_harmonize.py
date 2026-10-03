from datetime import datetime, timedelta

import numpy as np
import pytest

from antarctic_routing.config import DomainSection
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.preprocessing.harmonize import (
    area_mean_regrid,
    fill_missing,
    normalize_concentration,
    regrid_latlon,
    regrid_vector_latlon,
    to_dataset,
)

DOMAIN = DomainSection(lat_min=-66.0, lat_max=-55.0, lon_min=-72.0, lon_max=-52.0)


@pytest.fixture(scope="module")
def grid():
    return PolarGrid.from_domain(DOMAIN, resolution_km=25)


def _src_axes():
    lat = np.arange(-45.0, -75.25, -0.25)  # ERA5-style descending latitude
    lon = np.arange(-85.0, -34.75, 0.25)
    return lat, lon


def test_linear_field_is_reproduced_by_bilinear_regrid(grid):
    lat, lon = _src_axes()
    field = 2.0 * lat[:, None] + 0.5 * lon[None, :]  # exactly linear in lat/lon
    out = regrid_latlon(field, lat, lon, grid, method="linear")
    expected = 2.0 * grid.lat2d + 0.5 * grid.lon2d
    assert np.allclose(out, expected, atol=1e-9)


def test_nearest_regrid_keeps_categories(grid):
    lat, lon = _src_axes()
    mask = (lon[None, :] > -62.0).astype(np.uint8) * np.ones((lat.size, 1), np.uint8)
    out = regrid_latlon(mask, lat, lon, grid, method="nearest")
    assert set(np.unique(out)) <= {0, 1}


def test_points_outside_source_become_nan(grid):
    lat = np.arange(-60.0, -55.0, 0.5)
    lon = np.arange(-72.0, -52.0, 0.5)
    out = regrid_latlon(np.ones((lat.size, lon.size)), lat, lon, grid)
    assert np.isnan(out).any() and np.isfinite(out).any()


def test_uniform_eastward_wind_is_rotated_into_grid_components(grid):
    lat, lon = _src_axes()
    u = np.full((lat.size, lon.size), 10.0)
    v = np.zeros_like(u)
    ux, uy = regrid_vector_latlon(u, v, lat, lon, grid)
    lam = np.deg2rad(grid.lon2d)
    assert np.allclose(ux, 10.0 * np.cos(lam))
    assert np.allclose(uy, -10.0 * np.sin(lam))
    assert np.allclose(np.hypot(ux, uy), 10.0)


def test_area_mean_regrid_conserves_mean_of_fine_cells():
    fine = PolarGrid(x=np.arange(5) * 1000.0 + 500, y=np.arange(4) * 1000.0 + 500, resolution_m=1000)
    coarse = PolarGrid(x=np.array([1000.0, 3000.0]), y=np.array([1000.0, 3000.0]), resolution_m=2000)
    values = np.arange(20, dtype=float).reshape(4, 5)
    values[0, 0] = np.nan  # missing source cell is ignored, not treated as 0
    out = area_mean_regrid(values, fine.x, fine.y, coarse)
    assert out[0, 0] == pytest.approx(np.nanmean([np.nan, 1, 5, 6]))
    assert out[1, 1] == pytest.approx(np.mean([12, 13, 17, 18]))


def test_fill_missing_uses_nearest_valid_and_flags_imputed():
    values = np.array([[0.1, np.nan, 0.3], [0.4, 0.5, np.nan]])
    ocean = np.array([[True, True, True], [True, True, False]])  # last cell is land
    filled, imputed = fill_missing(values, ocean)
    assert imputed.tolist() == [[False, True, False], [False, False, False]]
    assert filled[0, 1] in (0.1, 0.3, 0.5)
    assert np.isnan(filled[1, 2])  # land stays masked, never imputed


def test_normalize_percent_concentration_to_fraction():
    out, flags = normalize_concentration(np.array([0.0, 50.0, 100.0, 104.0, np.nan]), units="percent")
    assert np.allclose(out[:3], [0.0, 0.5, 1.0])
    assert out[3] == 1.0 and flags["clipped"] == 1
    assert np.isnan(out[4])


def test_normalize_rejects_unknown_units():
    with pytest.raises(ValueError):
        normalize_concentration(np.zeros(2), units="okta")


def test_to_dataset_has_polar_coords_and_crs(grid):
    times = [datetime(2026, 11, 1) + timedelta(days=i) for i in range(3)]
    data = np.zeros((3, *grid.shape))
    ds = to_dataset(grid, times, ice_concentration=data, land_mask=np.zeros(grid.shape, bool))
    assert ds.attrs["crs"] == "EPSG:3031"
    assert ds["ice_concentration"].dims == ("time", "y", "x")
    assert ds["land_mask"].dims == ("y", "x")
    assert ds.sizes["time"] == 3
