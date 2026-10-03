"""Stage 6 - vessel speed and relative fuel model.

    Speed through water in ice:  v_s = max(v_cruise (1 - k C), v_min)
    Ground speed along an edge:  v_g = v_s + u_parallel
    Edge travel time:            dt  = d / v_g   (impassable if v_g <= 0)
    Relative fuel index:         c   = d [1 + lambda g(C)]

``g`` is quadratic (C^2) or piecewise (C^2 plus a steep linear term above the
vessel limit tau_v). This is a transparent *relative* index, not litres of fuel:
replace it with a validated performance curve when vessel data is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from antarctic_routing.config import ProjectConfig

PIECEWISE_SLOPE = 10.0


def ice_penalty(conc, kind: str = "quadratic", tau: float = 0.15):
    c = np.asarray(conc, float)
    if kind == "quadratic":
        g = c**2
    elif kind == "piecewise":
        g = c**2 + PIECEWISE_SLOPE * np.maximum(0.0, c - tau)
    else:
        raise ValueError("kind must be 'quadratic' or 'piecewise'")
    return float(g) if g.ndim == 0 else g


def fuel_index(distance_km, conc, lam: float, kind: str = "quadratic", tau: float = 0.15):
    out = np.asarray(distance_km, float) * (1.0 + lam * np.asarray(ice_penalty(conc, kind, tau)))
    return float(out) if out.ndim == 0 else out


def speed_in_ice_kmh(v_cruise: float, conc, k: float, v_min: float):
    out = np.maximum(v_cruise * (1.0 - k * np.asarray(conc, float)), v_min)
    return float(out) if out.ndim == 0 else out


def edge_travel_hours(distance_km: float, v_water_kmh: float, u_parallel_kmh: float) -> float:
    v_ground = v_water_kmh + u_parallel_kmh
    return float("inf") if v_ground <= 0 else distance_km / v_ground


@dataclass(frozen=True)
class VesselModel:
    cruise_kmh: float
    min_kmh: float
    ice_k: float
    tau: float
    lam: float
    penalty: str = "quadratic"

    def __post_init__(self) -> None:
        if not 0 < self.min_kmh <= self.cruise_kmh:
            raise ValueError("require 0 < min_kmh <= cruise_kmh")

    @classmethod
    def from_config(cls, cfg: ProjectConfig) -> VesselModel:
        return cls(
            cruise_kmh=cfg.vessel.cruise_speed_kmh,
            min_kmh=cfg.vessel.min_speed_kmh,
            ice_k=cfg.vessel.ice_speed_reduction,
            tau=cfg.vessel.max_ice_concentration,
            lam=cfg.routing.fuel_penalty_lambda,
            penalty=cfg.routing.fuel_penalty,
        )
