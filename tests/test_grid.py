import numpy as np
import pytest
from pyproj import Geod, Transformer

from antarctic_routing.config import DomainSection
from antarctic_routing.preprocessing.grid import PolarGrid

DOMAIN = DomainSection(lat_min=-66.0, lat_max=-55.0, lon_min=-72.0, lon_max=-52.0)


@pytest.fixture(scope="module")
def grid():
    return PolarGrid.from_domain(DOMAIN, resolution_km=10)


def test_epsg3031_orientation_greenwich_points_up():
    t = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    x0, y0 = t.transform(0.0, -70.0)
    x90, y90 = t.transform(90.0, -70.0)
    assert abs(x0) < 1e-6 and y0 > 0
    assert x90 > 0 and abs(y90) < 1e-6


def test_grid_spacing_and_shape(grid):
    assert grid.resolution_m == 10_000
    assert np.allclose(np.diff(grid.x), 10_000)
    assert np.allclose(np.diff(grid.y), 10_000)
    assert grid.shape == (grid.y.size, grid.x.size)
    assert grid.lat2d.shape == grid.shape


def test_grid_covers_entire_domain_boundary(grid):
    lats = np.r_[np.linspace(-66, -55, 50), np.full(50, -66.0), np.full(50, -55.0)]
    lons = np.r_[np.full(50, -72.0), np.linspace(-72, -52, 50), np.linspace(-72, -52, 50)]
    for lat, lon in zip(lats, lons):
        grid.cell_of(lat, lon)  # must not raise


def test_cell_of_returns_nearest_centre(grid):
    row, col = grid.cell_of(-63.0, -59.0)
    x, y = grid.to_xy(-63.0, -59.0)
    assert abs(grid.x[col] - x) <= 5_000 + 1e-6
    assert abs(grid.y[row] - y) <= 5_000 + 1e-6


def test_point_outside_grid_raises(grid):
    with pytest.raises(ValueError, match="outside"):
        grid.cell_of(-40.0, 0.0)


def test_xy_latlon_roundtrip(grid):
    lat, lon = grid.to_latlon(*grid.to_xy(-60.25, -61.5))
    assert lat == pytest.approx(-60.25, abs=1e-9)
    assert lon == pytest.approx(-61.5, abs=1e-9)


@pytest.mark.parametrize("lon", [0.0, 90.0, -60.0, 135.0])
def test_vector_rotation_matches_projection_finite_difference(grid, lon):
    lat = -62.0
    t = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    geod = Geod(ellps="WGS84")
    x0, y0 = t.transform(lon, lat)
    for azimuth, (ue, un) in ((90.0, (1.0, 0.0)), (0.0, (0.0, 1.0))):
        lon1, lat1, _ = geod.fwd(lon, lat, azimuth, 10.0)  # move 10 m east/north
        x1, y1 = t.transform(lon1, lat1)
        expected = np.array([x1 - x0, y1 - y0])
        expected /= np.linalg.norm(expected)
        ux, uy = grid.rotate_en_to_xy(np.array(ue), np.array(un), np.array(lon))
        assert float(ux) == pytest.approx(expected[0], abs=1e-4)
        assert float(uy) == pytest.approx(expected[1], abs=1e-4)


def test_rotation_preserves_vector_magnitude(grid):
    ue = np.array([3.0, -1.0])
    un = np.array([4.0, 2.0])
    ux, uy = grid.rotate_en_to_xy(ue, un, np.array([-65.0, 12.0]))
    assert np.allclose(np.hypot(ux, uy), np.hypot(ue, un))


def test_rotation_inverse_roundtrip(grid):
    ue, un, lon = np.array([1.5]), np.array([-0.5]), np.array([-58.0])
    back = grid.rotate_xy_to_en(*grid.rotate_en_to_xy(ue, un, lon), lon)
    assert np.allclose(back, (ue, un))


def test_projected_spacing_overstates_ground_distance_north_of_71s(grid):
    # EPSG:3031 has true scale at 71S, so 10 km of grid spans < 10 km on the ground at 60S.
    row, col = grid.cell_of(-60.0, -60.0)
    d = grid.geodesic_distance_m(row, col, row, col + 1)
    assert 9_000 < d < 10_000
