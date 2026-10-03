"""Controlled-synthetic environmental scenarios for the Drake Passage sector.

This module builds a *schematic* world for developing and testing the routing
engine before real NSIDC/OSI SAF, ERA5 and CMEMS data are wired in:

* a schematic land mask (Tierra del Fuego / Cape Horn, South Shetland Islands,
  Antarctic Peninsula) - hand-drawn polygons, NOT a navigational coastline;
* a seasonally retreating marginal ice zone with a Weddell-Sea ice tongue;
* an ensemble of joint scenarios whose ice-edge shift and spatially coherent
  concentration anomalies grow with lead time;
* an eastward Antarctic Circumpolar Current through the Drake Passage.

Everything returned is labelled ``execution_mode="controlled_synthetic"`` so it
can never be mistaken for real observations or forecasts.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.ndimage import gaussian_filter

from antarctic_routing.config import ProjectConfig
from antarctic_routing.preprocessing.climatology import day_of_season
from antarctic_routing.preprocessing.grid import PolarGrid

# (lat, lon) vertices - schematic only.
_TIERRA_DEL_FUEGO = [
    (-54.0, -72.5), (-54.0, -65.0), (-55.0, -65.2), (-55.3, -66.3), (-55.7, -67.0),
    (-56.0, -67.3), (-55.9, -68.6), (-55.5, -70.5), (-55.2, -72.5),
]
_PENINSULA = [
    (-63.2, -56.8), (-63.6, -58.4), (-64.2, -60.6), (-64.8, -62.6), (-65.4, -63.9),
    (-66.0, -65.5), (-67.5, -67.5), (-70.5, -68.5), (-70.5, -62.0), (-67.0, -62.5),
    (-66.0, -61.5), (-65.0, -60.3), (-64.2, -58.8), (-63.6, -57.3),
]
# (centre_lat, centre_lon, semi_axis_lat_deg, semi_axis_lon_deg)
_ISLANDS = [
    (-61.15, -55.20, 0.10, 0.45),  # Elephant Island
    (-62.10, -58.40, 0.12, 0.55),  # King George Island
    (-62.35, -59.45, 0.08, 0.35),  # Nelson / Robert / Greenwich group
    (-62.60, -60.40, 0.10, 0.45),  # Livingston Island
    (-62.95, -60.62, 0.05, 0.12),  # Deception Island
    (-62.80, -61.30, 0.06, 0.25),  # Snow / Smith Islands
]


def schematic_land_mask(grid: PolarGrid) -> np.ndarray:
    """Boolean land mask (True = land) on ``grid`` from schematic shapes."""
    pts = np.column_stack([grid.lat2d.ravel(), grid.lon2d.ravel()])
    land = np.zeros(pts.shape[0], bool)
    for poly in (_TIERRA_DEL_FUEGO, _PENINSULA):
        land |= MplPath(poly).contains_points(pts)
    for clat, clon, a, b in _ISLANDS:
        land |= ((pts[:, 0] - clat) / a) ** 2 + ((pts[:, 1] - clon) / b) ** 2 <= 1.0
    return land.reshape(grid.shape)


@dataclass
class ScenarioSet:
    """Joint environmental scenarios on the polar grid.

    ``conc`` is (K, T, ny, nx) sea-ice concentration (fraction, NaN on land);
    currents are grid-aligned x/y components in m/s, shape (T, ny, nx).
    """

    grid: PolarGrid
    start: datetime
    time_step_hours: float
    land: np.ndarray
    conc: np.ndarray
    current_x: np.ndarray
    current_y: np.ndarray
    execution_mode: str
    description: str
    truth_conc: np.ndarray | None = None
    berg: np.ndarray | None = None  # (K, T, ny, nx) bool, optional

    @property
    def n_scenarios(self) -> int:
        return self.conc.shape[0]

    @property
    def n_times(self) -> int:
        return self.conc.shape[1]

    @property
    def horizon_hours(self) -> float:
        return self.n_times * self.time_step_hours

    @property
    def ocean(self) -> np.ndarray:
        return ~self.land

    def time_index(self, hours: float) -> int:
        """Index of the layer valid at ``hours`` after start (floor, clamped)."""
        return int(min(max(hours // self.time_step_hours, 0), self.n_times - 1))

    def mean_conc(self) -> np.ndarray:
        """Ensemble-mean concentration (T, ny, nx); NaN on land."""
        out = np.full(self.conc.shape[1:], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN missing ocean cells
            out[:, self.ocean] = np.nanmean(self.conc[:, :, self.ocean], axis=0)
        return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _edge_latitude(dos: int, lon: np.ndarray) -> np.ndarray:
    """Schematic mean ice-edge latitude for a day of season (0 = 1 Nov)."""
    retreat = -61.6 - 0.06 * dos
    wave = 0.35 * np.sin(np.deg2rad(lon) * 8.0)
    weddell = 1.0 * _sigmoid((lon + 58.5) / 1.0)  # ice tongue east of the Peninsula
    return retreat + wave + weddell


def _tongues(lon: np.ndarray, phase: float) -> np.ndarray:
    """Ice tongues/embayments along the edge (~9 deg wavelength).

    Their along-edge position (``phase``) differs between scenarios, which is
    what makes route choice a genuine trade-off under uncertainty.
    """
    return 0.55 * np.sin(np.deg2rad(lon) * 40.0 + phase)


def _member(
    rng: np.random.Generator,
    base_edges: list[np.ndarray],
    lat_lon: tuple[np.ndarray, np.ndarray],
    land: np.ndarray,
    smooth_cells: float,
) -> np.ndarray:
    n_t = len(base_edges)
    lat, lon = lat_lon
    tongues = _tongues(lon, rng.normal(0.0, 0.6))
    shift_amp = rng.normal()
    eta = np.zeros(lat.shape)
    out = np.empty((n_t, *lat.shape), np.float32)
    walk = 0.0
    for t in range(n_t):
        walk += rng.normal(0, 0.05)
        shift = shift_amp * (0.15 + 0.08 * t) + walk
        noise = gaussian_filter(rng.normal(size=lat.shape), smooth_cells)
        noise /= noise.std() + 1e-12
        eta = 0.8 * eta + 0.6 * noise
        amp = 0.06 * (1 + 0.3 * t)
        c = 0.95 * _sigmoid((base_edges[t] + tongues + shift - lat) / 0.35)
        c = np.clip(c + amp * eta * (0.3 + 2.8 * c * (1 - c)), 0.0, 1.0)
        c[land] = np.nan
        out[t] = c
    return out


def generate_synthetic(
    cfg: ProjectConfig,
    grid: PolarGrid,
    start: date,
    n_days: int,
    n_scenarios: int,
    seed: int = 0,
) -> ScenarioSet:
    """Build ``n_scenarios`` joint scenarios of ``n_days`` daily layers from ``start``."""
    months = cfg.project.season_months
    if n_days < 1 or n_scenarios < 1:
        raise ValueError("n_days and n_scenarios must be positive")
    days = [start + timedelta(days=t) for t in range(n_days)]
    try:
        dos = [day_of_season(d, months) for d in days]
    except ValueError as exc:
        raise ValueError(f"synthetic start {start} is outside the configured season: {exc}") from None

    land = schematic_land_mask(grid)
    lat, lon = grid.lat2d, grid.lon2d
    base_edges = [_edge_latitude(d, lon) for d in dos]
    smooth = 60_000.0 / grid.resolution_m
    rng = np.random.default_rng(seed)
    conc = np.stack([_member(rng, base_edges, (lat, lon), land, smooth) for _ in range(n_scenarios)])
    truth = _member(np.random.default_rng([seed, 99991]), base_edges, (lat, lon), land, smooth)

    # ACC: eastward jet centred near 58S, weak meridional meander. m/s.
    u_e = 0.05 + 0.35 * np.exp(-(((lat + 58.0) / 2.5) ** 2))
    v_n = 0.04 * np.sin(np.deg2rad(lon) * 6.0)
    cx, cy = grid.rotate_en_to_xy(u_e, v_n, lon)
    cx[land] = 0.0
    cy[land] = 0.0

    return ScenarioSet(
        grid=grid,
        start=datetime(start.year, start.month, start.day),
        time_step_hours=cfg.grid.time_step_hours,
        land=land,
        conc=conc,
        current_x=np.broadcast_to(cx, (n_days, *grid.shape)).copy(),
        current_y=np.broadcast_to(cy, (n_days, *grid.shape)).copy(),
        execution_mode="controlled_synthetic",
        description=(
            "Schematic Drake Passage/Bransfield Strait world with a retreating ice edge; "
            f"{n_scenarios} joint scenarios, seed={seed}. Not real data."
        ),
        truth_conc=truth,
    )
