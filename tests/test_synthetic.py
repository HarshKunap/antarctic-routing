from datetime import date
from pathlib import Path

import numpy as np
import pytest

from antarctic_routing.config import load_config
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.synthetic import generate_synthetic, schematic_land_mask


@pytest.fixture(scope="module")
def cfg():
    return load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")


@pytest.fixture(scope="module")
def grid(cfg):
    return PolarGrid.from_domain(cfg.domain, resolution_km=20)


@pytest.fixture(scope="module")
def scen(cfg, grid):
    return generate_synthetic(cfg, grid, start=date(2026, 11, 20), n_days=5, n_scenarios=30, seed=7)


def test_route_endpoints_are_ocean(cfg, grid):
    land = schematic_land_mask(grid)
    for p in (cfg.route.origin, cfg.route.destination):
        assert not land[grid.cell_of(p.lat, p.lon)]


@pytest.mark.parametrize("lat,lon", [(-62.1, -58.4), (-64.8, -61.5), (-55.6, -68.5)])
def test_known_land_points_are_land(grid, lat, lon):
    assert schematic_land_mask(grid)[grid.cell_of(lat, lon)]


def test_shapes_ranges_and_label(scen, grid):
    assert scen.conc.shape == (30, 5, *grid.shape)
    ocean = scen.conc[:, :, ~scen.land]
    assert np.nanmin(ocean) >= 0 and np.nanmax(ocean) <= 1
    assert np.isnan(scen.conc[:, :, scen.land]).all()
    assert scen.execution_mode == "controlled_synthetic"
    assert scen.truth_conc.shape == (5, *grid.shape)


def test_ice_increases_towards_the_pole(scen, grid):
    mean = scen.conc[:, 0][:, scen.ocean].mean(axis=0)
    lat = grid.lat2d[scen.ocean]
    assert mean[lat < -65].mean() > mean[lat > -58].mean() + 0.3


def test_ice_edge_retreats_through_the_season(cfg, grid):
    early = generate_synthetic(cfg, grid, date(2026, 11, 15), 1, 10, seed=1)
    late = generate_synthetic(cfg, grid, date(2027, 1, 15), 1, 10, seed=1)
    assert np.nanmean(late.conc) < np.nanmean(early.conc)


def test_ensemble_spread_grows_with_lead_time(scen):
    spread = [float(scen.conc[:, t][:, scen.ocean].std(axis=0).mean()) for t in range(scen.n_times)]
    assert spread[-1] > spread[0] > 0


def test_generation_is_deterministic_for_a_seed(cfg, grid):
    a = generate_synthetic(cfg, grid, date(2026, 11, 20), 2, 4, seed=3)
    b = generate_synthetic(cfg, grid, date(2026, 11, 20), 2, 4, seed=3)
    assert np.array_equal(a.conc, b.conc, equal_nan=True)


def test_drake_passage_current_flows_east(scen, grid):
    r, c = grid.cell_of(-58.0, -64.0)
    ue, _ = grid.rotate_xy_to_en(scen.current_x[0, r, c], scen.current_y[0, r, c], grid.lon2d[r, c])
    assert ue > 0.1


def test_time_index_floor_and_clamp(scen):
    assert scen.time_index(0.0) == 0
    assert scen.time_index(23.9) == 0
    assert scen.time_index(24.0) == 1
    assert scen.time_index(1e6) == scen.n_times - 1


def test_start_outside_season_rejected(cfg, grid):
    with pytest.raises(ValueError, match="season"):
        generate_synthetic(cfg, grid, date(2026, 6, 1), 2, 2, seed=0)
