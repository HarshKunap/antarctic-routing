"""Stage 7 - static navigation graph over the polar grid.

Nodes are navigable cells. Edges use 8 or 16 neighbour moves; 16-connectivity
adds "knight" moves that allow headings closer to the true bearing and reduce
the zig-zag of pure 8-connected paths.

A move is valid only if every cell the straight segment passes through is
navigable, so routes cannot cut land corners. Edge lengths are geodesic (WGS84)
because projected distances are distorted away from 71S.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pyproj import Geod

from antarctic_routing.preprocessing.grid import PolarGrid

_GEOD = Geod(ellps="WGS84")

MOVES_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
MOVES_16 = MOVES_8 + [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1),
]


def _intermediate_cells(dr: int, dc: int) -> list[tuple[int, int]]:
    """Cells (relative offsets) a straight move crosses besides its endpoints."""
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    if abs(dr) == 1 and abs(dc) == 1:
        return [(sr, 0), (0, sc)]
    if abs(dr) == 1 and abs(dc) == 2:
        return [(0, sc), (sr, sc)]
    if abs(dr) == 2 and abs(dc) == 1:
        return [(sr, 0), (sr, sc)]
    return []


@dataclass
class RoutingGrid:
    grid: PolarGrid
    navigable: np.ndarray
    moves: list[tuple[int, int]]
    # adjacency[node] -> list of (neighbour, distance_km, unit_x, unit_y)
    adjacency: list[list[tuple[int, float, float, float]]] = field(repr=False)

    @property
    def nx(self) -> int:
        return self.grid.x.size

    def node(self, cell: tuple[int, int]) -> int:
        return cell[0] * self.nx + cell[1]

    def cell(self, node: int) -> tuple[int, int]:
        return divmod(node, self.nx)

    def edge_valid(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        target = self.node(b)
        return any(n == target for n, *_ in self.adjacency[self.node(a)])

    @classmethod
    def build(cls, grid: PolarGrid, navigable: np.ndarray, connectivity: int = 16) -> RoutingGrid:
        if connectivity not in (8, 16):
            raise ValueError("connectivity must be 8 or 16")
        nav = np.asarray(navigable, bool)
        ny, nx = nav.shape
        moves = MOVES_8 if connectivity == 8 else MOVES_16
        rows, cols = np.indices((ny, nx))
        adjacency: list[list[tuple[int, float, float, float]]] = [[] for _ in range(ny * nx)]
        lat, lon = grid.lat2d, grid.lon2d

        def nav_at(r: np.ndarray, c: np.ndarray) -> np.ndarray:
            inside = (r >= 0) & (r < ny) & (c >= 0) & (c < nx)
            out = np.zeros(r.shape, bool)
            out[inside] = nav[r[inside], c[inside]]
            return out

        for dr, dc in moves:
            r2, c2 = rows + dr, cols + dc
            ok = nav & nav_at(r2, c2)
            for ir, ic in _intermediate_cells(dr, dc):
                ok &= nav_at(rows + ir, cols + ic)
            src_r, src_c = rows[ok], cols[ok]
            dst_r, dst_c = r2[ok], c2[ok]
            _, _, dist_m = _GEOD.inv(lon[src_r, src_c], lat[src_r, src_c], lon[dst_r, dst_c], lat[dst_r, dst_c])
            norm = float(np.hypot(dr, dc))
            ux, uy = dc / norm, dr / norm  # columns follow +x, rows follow +y
            src = (src_r * nx + src_c).tolist()
            dst = (dst_r * nx + dst_c).tolist()
            for s, d, km in zip(src, dst, (np.asarray(dist_m) / 1000.0).tolist()):
                adjacency[s].append((d, km, ux, uy))
        return cls(grid=grid, navigable=nav, moves=list(moves), adjacency=adjacency)
