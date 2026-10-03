"""Common Antarctic polar-stereographic analysis grid (EPSG:3031).

EPSG:3031 is a south polar stereographic projection with true scale at 71S and
the Greenwich meridian pointing "up" (+y). In this projection

    x = rho * sin(lon),   y = rho * cos(lon)

where rho grows towards the equator. East/north vector components (winds,
currents) must therefore be *rotated* by the longitude before being used as
grid x/y components - interpolating them as if east were +x is a classic bug.

Because the projection is not equal-distance away from 71S, ground distances
between cells are computed geodesically, never from projected spacing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from pyproj import Geod, Transformer

from antarctic_routing.config import DomainSection

CRS = "EPSG:3031"
_TO_XY = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
_TO_LL = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class PolarGrid:
    """Regular grid of cell centres in EPSG:3031 metres (x, y ascending)."""

    x: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    resolution_m: float

    @classmethod
    def from_domain(cls, domain: DomainSection, resolution_km: float, samples: int = 200) -> PolarGrid:
        """Smallest resolution-aligned grid enclosing the lat/lon domain box.

        The box edges are densely sampled because lat/lon lines are curved in
        polar stereographic space - projecting only the four corners would clip
        the domain.
        """
        res = resolution_km * 1000.0
        lat = np.linspace(domain.lat_min, domain.lat_max, samples)
        lon = np.linspace(domain.lon_min, domain.lon_max, samples)
        b_lat = np.r_[lat, lat, np.full(samples, domain.lat_min), np.full(samples, domain.lat_max)]
        b_lon = np.r_[
            np.full(samples, domain.lon_min), np.full(samples, domain.lon_max), lon, lon
        ]
        bx, by = _TO_XY.transform(b_lon, b_lat)
        x_edges = np.arange(np.floor(bx.min() / res) * res, np.ceil(bx.max() / res) * res + res / 2, res)
        y_edges = np.arange(np.floor(by.min() / res) * res, np.ceil(by.max() / res) * res + res / 2, res)
        return cls(x=x_edges[:-1] + res / 2, y=y_edges[:-1] + res / 2, resolution_m=res)

    # ------------------------------------------------------------------ shape
    @property
    def shape(self) -> tuple[int, int]:
        return (self.y.size, self.x.size)

    @cached_property
    def _xy2d(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.y)

    @cached_property
    def _latlon2d(self) -> tuple[np.ndarray, np.ndarray]:
        xx, yy = self._xy2d
        lon, lat = _TO_LL.transform(xx, yy)
        return lat, lon

    @property
    def lat2d(self) -> np.ndarray:
        return self._latlon2d[0]

    @property
    def lon2d(self) -> np.ndarray:
        return self._latlon2d[1]

    # ------------------------------------------------------- transformations
    @staticmethod
    def to_xy(lat, lon):
        return _TO_XY.transform(lon, lat)

    @staticmethod
    def to_latlon(x, y):
        lon, lat = _TO_LL.transform(x, y)
        return lat, lon

    def cell_of_xy(self, x: float, y: float) -> tuple[int, int]:
        res = self.resolution_m
        col = int(np.floor((x - (self.x[0] - res / 2)) / res))
        row = int(np.floor((y - (self.y[0] - res / 2)) / res))
        if not (0 <= row < self.y.size and 0 <= col < self.x.size):
            raise ValueError(f"point ({x:.0f}, {y:.0f}) m is outside the grid")
        return row, col

    def cell_of(self, lat: float, lon: float) -> tuple[int, int]:
        """(row, col) of the cell containing a WGS84 point."""
        try:
            return self.cell_of_xy(*self.to_xy(lat, lon))
        except ValueError:
            raise ValueError(f"point ({lat}, {lon}) is outside the grid") from None

    def cell_latlon(self, row: int, col: int) -> tuple[float, float]:
        return float(self.lat2d[row, col]), float(self.lon2d[row, col])

    def geodesic_distance_m(self, r1: int, c1: int, r2: int, c2: int) -> float:
        lat1, lon1 = self.cell_latlon(r1, c1)
        lat2, lon2 = self.cell_latlon(r2, c2)
        return float(_GEOD.inv(lon1, lat1, lon2, lat2)[2])

    # ------------------------------------------------------ vector rotation
    @staticmethod
    def rotate_en_to_xy(u_east, u_north, lon_deg):
        """Rotate east/north components into EPSG:3031 grid x/y components."""
        lam = np.deg2rad(lon_deg)
        c, s = np.cos(lam), np.sin(lam)
        return u_east * c + u_north * s, -u_east * s + u_north * c

    @staticmethod
    def rotate_xy_to_en(u_x, u_y, lon_deg):
        """Inverse of :meth:`rotate_en_to_xy`."""
        lam = np.deg2rad(lon_deg)
        c, s = np.cos(lam), np.sin(lam)
        return u_x * c - u_y * s, u_x * s + u_y * c
