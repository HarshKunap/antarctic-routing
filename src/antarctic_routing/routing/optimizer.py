"""Stage 7 - time-dependent route search (A* / Dijkstra).

Edge costs depend on *when* the vessel traverses them: conditions are read from
the layer valid at departure from the edge's start node (speed, current) and at
arrival at its end node (ice for fuel and risk).

Generalised edge cost:

    cost = w_D d + w_T dt + w_F d[1 + lambda g(C_j)] + w_R (-ln(1 - p_haz_j))

The risk term is additive: summing -ln(1 - p) along a path equals
-ln P(no breach) under an independence approximation. It is used only to steer
the search; reported route risk always comes from joint-scenario evaluation.

Label-setting search is exact for travel time when the FIFO property holds
(leaving later never arrives earlier). With daily layers and generalised costs
it is a well-behaved heuristic; candidates are therefore always re-evaluated
across the full scenario set before selection.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from pyproj import Geod

from antarctic_routing.routing.fuel import PIECEWISE_SLOPE, VesselModel
from antarctic_routing.routing.graph import RoutingGrid
from antarctic_routing.routing.hazard import combine_independent, exceedance_probability
from antarctic_routing.synthetic import ScenarioSet

_GEOD = Geod(ellps="WGS84")
_P_CAP = 1.0 - 1e-9


class NoRouteError(RuntimeError):
    """No navigable path connects origin and destination."""


@dataclass(frozen=True)
class Objective:
    name: str
    w_distance: float = 0.0
    w_time: float = 0.0
    w_fuel: float = 0.0
    w_risk: float = 0.0


@dataclass
class EnvironmentLayers:
    """Deterministic time-indexed layers the search reads from."""

    conc: np.ndarray        # (T, ny, nx)
    p_haz: np.ndarray       # (T, ny, nx)
    current_x: np.ndarray   # (T, ny, nx) m/s
    current_y: np.ndarray
    time_step_hours: float

    @property
    def n_times(self) -> int:
        return self.conc.shape[0]

    @cached_property
    def flat(self) -> tuple[list, list, list, list]:
        def lists(a):
            return [layer.ravel().tolist() for layer in a]

        return lists(self.conc), lists(self.p_haz), lists(self.current_x), lists(self.current_y)

    @cached_property
    def max_current_kmh(self) -> float:
        return float(np.nanmax(np.hypot(self.current_x, self.current_y))) * 3.6


def layers_from_scenarios(world: ScenarioSet, tau: float) -> EnvironmentLayers:
    """Ensemble-mean concentration and ensemble hazard probability."""
    p_ice = exceedance_probability(world.conc, tau)
    p_haz = p_ice
    if world.berg is not None:
        p_haz = combine_independent(p_ice, world.berg.mean(axis=0))
    p_haz = np.where(world.land[None], 1.0, p_haz)
    return EnvironmentLayers(world.mean_conc(), p_haz, world.current_x, world.current_y, world.time_step_hours)


def layers_for_scenario(world: ScenarioSet, k: int, tau: float) -> EnvironmentLayers:
    """Layers for a single joint scenario (hazard is a 0/1 indicator)."""
    conc = world.conc[k]
    hit = np.nan_to_num(conc, nan=1.0) >= tau
    if world.berg is not None:
        hit |= world.berg[k]
    return EnvironmentLayers(conc, hit.astype(float), world.current_x, world.current_y, world.time_step_hours)


@dataclass
class Route:
    objective: str
    cells: list[tuple[int, int]]
    arrival_hours: list[float]
    cost: float
    distance_km: float
    meta: dict = field(default_factory=dict)


def plan_route(
    rgrid: RoutingGrid,
    layers: EnvironmentLayers,
    vessel: VesselModel,
    start: tuple[int, int],
    goal: tuple[int, int],
    objective: Objective,
    use_heuristic: bool = True,
    depart_hours: float = 0.0,
) -> Route:
    nav = rgrid.navigable
    for label, cell in (("origin", start), ("destination", goal)):
        if not nav[cell]:
            raise NoRouteError(f"{label} cell {cell} is not navigable")

    conc_l, p_l, cx_l, cy_l = layers.flat
    step, last = layers.time_step_hours, layers.n_times - 1
    wD, wT, wF, wR = objective.w_distance, objective.w_time, objective.w_fuel, objective.w_risk
    v_cruise, v_min, k_ice = vessel.cruise_kmh, vessel.min_kmh, vessel.ice_k
    lam, tau, piecewise = vessel.lam, vessel.tau, vessel.penalty == "piecewise"

    s, g = rgrid.node(start), rgrid.node(goal)
    n_nodes = len(rgrid.adjacency)

    if use_heuristic:
        lat, lon = rgrid.grid.lat2d.ravel(), rgrid.grid.lon2d.ravel()
        glat, glon = lat[g], lon[g]
        _, _, d_goal = _GEOD.inv(lon, lat, np.full_like(lon, glon), np.full_like(lat, glat))
        v_max = v_cruise + layers.max_current_kmh
        h_scale = wD + wT / v_max + wF
        h = (np.asarray(d_goal) / 1000.0 * h_scale * (1 - 1e-12)).tolist()
    else:
        h = [0.0] * n_nodes

    best = [math.inf] * n_nodes
    time_at = [0.0] * n_nodes
    parent = [-1] * n_nodes
    best[s] = 0.0
    time_at[s] = depart_hours
    heap = [(h[s], 0.0, s)]
    closed = bytearray(n_nodes)

    while heap:
        _, cost, n = heapq.heappop(heap)
        if closed[n]:
            continue
        closed[n] = 1
        if n == g:
            break
        t = time_at[n]
        k = min(int(t // step), last)
        ck, cxk, cyk = conc_l[k], cx_l[k], cy_l[k]
        ci = ck[n]
        ci = 1.0 if ci != ci else ci
        for n2, d, ux, uy in rgrid.adjacency[n]:
            if closed[n2]:
                continue
            cj0 = ck[n2]
            cj0 = 1.0 if cj0 != cj0 else cj0
            vs = v_cruise * (1.0 - k_ice * 0.5 * (ci + cj0))
            if vs < v_min:
                vs = v_min
            vg = vs + (cxk[n] * ux + cyk[n] * uy) * 3.6
            if vg <= 0.0:
                continue
            dt = d / vg
            t2 = t + dt
            k2 = min(int(t2 // step), last)
            cj = conc_l[k2][n2]
            cj = 1.0 if cj != cj else cj
            pen = cj * cj + (PIECEWISE_SLOPE * (cj - tau) if piecewise and cj > tau else 0.0)
            p = p_l[k2][n2]
            p = 1.0 if p != p else (p if p < _P_CAP else _P_CAP)
            new = cost + wD * d + wT * dt + wF * d * (1.0 + lam * pen) - wR * math.log1p(-p)
            if new < best[n2]:
                best[n2] = new
                time_at[n2] = t2
                parent[n2] = n
                heapq.heappush(heap, (new + h[n2], new, n2))

    if not closed[g]:
        raise NoRouteError(f"no navigable route from {start} to {goal}")

    path = [g]
    while path[-1] != s:
        path.append(parent[path[-1]])
    path.reverse()
    cells = [rgrid.cell(n) for n in path]
    dist_lookup = {}
    for a, b in zip(path, path[1:]):
        dist_lookup[(a, b)] = next(d for n2, d, *_ in rgrid.adjacency[a] if n2 == b)
    return Route(
        objective=objective.name,
        cells=cells,
        arrival_hours=[time_at[n] - depart_hours for n in path],
        cost=best[g],
        distance_km=float(sum(dist_lookup.values())),
    )
