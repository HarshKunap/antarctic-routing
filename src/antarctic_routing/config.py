"""Stage 1 - scenario configuration, validated with Pydantic.

The configuration is the single source of truth for the operational scenario:
region, season, route endpoints, vessel profile, forecast window, grid and risk
budget. Validation rejects inconsistent or physically meaningless settings
before any data is downloaded or any model is run.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

KNOTS_TO_KMH = 1.852
_Z_95 = 1.959963984540054


def knots_to_kmh(knots: float) -> float:
    """v_km/h = 1.852 * v_knots."""
    return KNOTS_TO_KMH * knots


def _z_for_confidence(confidence: float) -> float:
    if math.isclose(confidence, 0.95):
        return _Z_95
    from scipy.stats import norm

    return float(norm.ppf(0.5 + confidence / 2))


def wilson_upper_bound(events: int, n: int, confidence: float = 0.95) -> float:
    """Upper limit of the two-sided Wilson score interval for a binomial proportion.

    Used to decide whether an estimated breach probability is *demonstrably*
    within the risk budget, given only a finite number of scenarios.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= events <= n:
        raise ValueError("events must be within [0, n]")
    z = _z_for_confidence(confidence)
    p = events / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return min(1.0, centre + half)


def min_scenarios_for_budget(risk_budget: float, confidence: float = 0.95) -> int:
    """Smallest scenario count for which zero breaches certifies ``risk_budget``.

    With zero events the Wilson upper bound is z^2 / (n + z^2), so
    n >= z^2 (1 - r) / r.
    """
    if not 0 < risk_budget < 1:
        raise ValueError("risk_budget must be within (0, 1)")
    z = _z_for_confidence(confidence)
    n = math.ceil(z**2 * (1 - risk_budget) / risk_budget)
    while wilson_upper_bound(0, n, confidence) > risk_budget:  # guard float edge
        n += 1
    return n


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GeoPoint(_Strict):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class ProjectSection(_Strict):
    name: str
    region: str
    season_months: list[int] = Field(min_length=1, max_length=12)

    @field_validator("season_months")
    @classmethod
    def _months(cls, months: list[int]) -> list[int]:
        if any(not 1 <= m <= 12 for m in months):
            raise ValueError("season_months must be within 1..12")
        if len(set(months)) != len(months):
            raise ValueError("season_months must be unique")
        return months


class DomainSection(_Strict):
    crs: Literal["EPSG:3031"] = "EPSG:3031"
    lat_min: float = Field(ge=-90, le=-40)
    lat_max: float = Field(ge=-90, le=-40)
    lon_min: float = Field(ge=-180, le=180)
    lon_max: float = Field(ge=-180, le=180)

    @model_validator(mode="after")
    def _ordered(self) -> DomainSection:
        if self.lat_min >= self.lat_max:
            raise ValueError("lat_min must be < lat_max")
        if self.lon_min >= self.lon_max:
            raise ValueError("lon_min must be < lon_max (antimeridian domains unsupported)")
        return self

    def contains(self, point: GeoPoint) -> bool:
        return (
            self.lat_min <= point.lat <= self.lat_max
            and self.lon_min <= point.lon <= self.lon_max
        )


class RouteSection(_Strict):
    origin: GeoPoint
    destination: GeoPoint
    departure_start: date
    departure_end: date

    @model_validator(mode="after")
    def _window(self) -> RouteSection:
        if self.departure_start > self.departure_end:
            raise ValueError("departure_start must not be after departure_end")
        return self


class GridSection(_Strict):
    resolution_km: float = Field(gt=0, le=100)
    time_step_hours: float = Field(gt=0, le=24 * 7)


class ForecastSection(_Strict):
    history_days: int = Field(ge=1)
    lead_days: int = Field(ge=1)
    ensemble_size: int = Field(ge=1)
    route_scenarios: int = Field(ge=1)


class VesselSection(_Strict):
    name: str
    ice_class: str
    cruise_speed_knots: float = Field(gt=0, le=40)
    max_ice_concentration: float = Field(gt=0, le=1)
    ice_speed_reduction: float = Field(ge=0, le=1)
    min_speed_knots: float = Field(gt=0)

    @property
    def cruise_speed_kmh(self) -> float:
        return knots_to_kmh(self.cruise_speed_knots)

    @property
    def min_speed_kmh(self) -> float:
        return knots_to_kmh(self.min_speed_knots)

    @model_validator(mode="after")
    def _speeds(self) -> VesselSection:
        if self.min_speed_knots > self.cruise_speed_knots:
            raise ValueError("min_speed_knots must not exceed cruise_speed_knots")
        return self


class RoutingSection(_Strict):
    risk_budget: float = Field(gt=0, lt=1)
    risk_definition: Literal["probability_of_any_defined_breach"]
    risk_estimator: Literal["wilson_upper", "point"] = "wilson_upper"
    confidence: float = Field(default=0.95, gt=0.5, lt=1)
    fuel_model: Literal["relative_index_v1"] = "relative_index_v1"
    fuel_penalty_lambda: float = Field(ge=0)
    fuel_penalty: Literal["quadratic", "piecewise"] = "quadratic"
    connectivity: Literal[8, 16] = 16
    risk_weights: list[float] = Field(min_length=1)

    @field_validator("risk_weights")
    @classmethod
    def _weights(cls, weights: list[float]) -> list[float]:
        if any(w < 0 for w in weights):
            raise ValueError("risk_weights must be non-negative")
        return sorted(set(weights))


class ProjectConfig(_Strict):
    project: ProjectSection
    domain: DomainSection
    route: RouteSection
    grid: GridSection
    forecast: ForecastSection
    vessel: VesselSection
    routing: RoutingSection

    @model_validator(mode="after")
    def _consistency(self) -> ProjectConfig:
        for label, point in (("origin", self.route.origin), ("destination", self.route.destination)):
            if not self.domain.contains(point):
                raise ValueError(f"route {label} ({point.lat}, {point.lon}) is outside the domain")
        if self.routing.risk_estimator == "wilson_upper":
            needed = min_scenarios_for_budget(self.routing.risk_budget, self.routing.confidence)
            if self.forecast.route_scenarios < needed:
                raise ValueError(
                    f"forecast.route_scenarios={self.forecast.route_scenarios} can never certify "
                    f"risk_budget={self.routing.risk_budget} at {self.routing.confidence:.0%} "
                    f"confidence; at least {needed} scenarios are required"
                )
        return self


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a YAML scenario configuration."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ProjectConfig.model_validate(raw)
