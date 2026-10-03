"""Small hand-built worlds shared by routing tests."""

from datetime import datetime

import numpy as np

from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.routing.fuel import VesselModel
from antarctic_routing.routing.optimizer import Objective
from antarctic_routing.synthetic import ScenarioSet

NY, NX, RES = 11, 21, 10_000.0
A, B = (5, 0), (5, 20)
DIST = Objective("shortest_distance", w_distance=1.0)
TIME = Objective("minimum_time", w_time=1.0)


def _grid() -> PolarGrid:
    x0, y0 = PolarGrid.to_xy(-62.0, -60.0)
    x0, y0 = round(x0 / RES) * RES, round(y0 / RES) * RES
    return PolarGrid(x=x0 + np.arange(NX) * RES, y=y0 + np.arange(NY) * RES, resolution_m=RES)


def _world(k=4, t=3, land=None, conc=None, cx=0.0, cy=0.0) -> ScenarioSet:
    grid = _grid()
    land = np.zeros((NY, NX), bool) if land is None else land
    c = np.zeros((k, t, NY, NX), np.float32) if conc is None else conc.astype(np.float32)
    c[:, :, land] = np.nan
    return ScenarioSet(
        grid=grid, start=datetime(2026, 12, 1), time_step_hours=24.0, land=land, conc=c,
        current_x=np.full((t, NY, NX), cx), current_y=np.full((t, NY, NX), cy),
        execution_mode="controlled_synthetic", description="unit-test world",
    )


def _vessel(knots=12.0, **kw) -> VesselModel:
    base = dict(cruise_kmh=knots * 1.852, min_kmh=min(3 * 1.852, knots * 1.852), ice_k=0.7,
                tau=0.15, lam=4.0, penalty="quadratic")
    base.update(kw)
    return VesselModel(**base)
