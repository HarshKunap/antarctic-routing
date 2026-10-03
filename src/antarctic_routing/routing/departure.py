"""Stage 8 - departure-window planning under a risk budget.

For every candidate departure date d, scenarios are generated from information
available at d (the ``scenario_factory`` is responsible for that), candidate
routes are planned and evaluated, and the date is flagged feasible when its
best route satisfies the risk budget:

    R*_d = argmin_R E[F_R | d]   s.t.   P(B_R | d) <= r_max

Selection rule (declared in advance): minimum expected fuel among feasible
dates, where dates within a relative fuel tolerance (default 1%) are treated as
equal and the earliest of them is chosen - sub-percent differences are below the
fuel model's resolution and must not push departures later. If no date is
feasible the sweep says so instead of recommending anything.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date

from antarctic_routing.routing.candidates import PlanResult, plan_candidates
from antarctic_routing.routing.fuel import VesselModel
from antarctic_routing.routing.graph import RoutingGrid
from antarctic_routing.synthetic import ScenarioSet

SELECTION_RULE = (
    "min expected fuel among feasible departure dates; dates within {tol:.0%} of the "
    "minimum are treated as equal and the earliest is selected"
)


def select_departure(options, fuel_tolerance: float = 0.01):
    """Apply the declared selection rule to feasible options (None if none)."""
    feasible = [o for o in options if o.feasible]
    if not feasible:
        return None
    best_fuel = min(o.expected_fuel for o in feasible)
    near = [o for o in feasible if o.expected_fuel <= best_fuel * (1 + fuel_tolerance)]
    return min(near, key=lambda o: o.departure)


@dataclass
class DepartureOption:
    departure: date
    status: str
    feasible: bool
    expected_hours: float
    expected_fuel: float
    p_breach: float
    p_breach_upper: float
    beyond_horizon_fraction: float
    route_labels: list[str]
    plan: PlanResult = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "departure": self.departure.isoformat(),
            "status": self.status,
            "feasible": self.feasible,
            "expected_hours": self.expected_hours,
            "expected_fuel": self.expected_fuel,
            "p_breach": self.p_breach,
            "p_breach_upper": self.p_breach_upper,
            "beyond_horizon_fraction": self.beyond_horizon_fraction,
            "route_labels": self.route_labels,
        }


@dataclass
class DepartureSweep:
    options: list[DepartureOption]
    selected: DepartureOption | None
    rule: str
    explanation: str

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "explanation": self.explanation,
            "selected": self.selected.departure.isoformat() if self.selected else None,
            "options": [o.to_dict() for o in self.options],
        }


def sweep_departures(
    dates: Sequence[date],
    scenario_factory: Callable[[date], ScenarioSet],
    vessel: VesselModel,
    origin: tuple[int, int],
    destination: tuple[int, int],
    risk_budget: float,
    risk_weights: Sequence[float],
    estimator: str = "wilson_upper",
    connectivity: int = 16,
    n_scenario_routes: int = 3,
    confidence: float = 0.95,
    seed: int = 0,
    fuel_tolerance: float = 0.01,
) -> DepartureSweep:
    options: list[DepartureOption] = []
    rgrid: RoutingGrid | None = None
    for d in dates:
        world = scenario_factory(d)
        if rgrid is None or rgrid.navigable.shape != world.ocean.shape or not (rgrid.navigable == world.ocean).all():
            rgrid = RoutingGrid.build(world.grid, world.ocean, connectivity)
        plan = plan_candidates(
            world, vessel, origin, destination, risk_budget, risk_weights, estimator,
            connectivity, n_scenario_routes, confidence, seed, rgrid=rgrid,
        )
        chosen = plan.recommended or next(
            (c for c in plan.candidates if "lowest_risk" in c.tags), None
        )
        if chosen is None:
            options.append(DepartureOption(d, plan.status, False, float("inf"), float("inf"),
                                           1.0, 1.0, 0.0, [], plan))
            continue
        ev = chosen.evaluation
        options.append(DepartureOption(
            d, plan.status, plan.status == "feasible", ev.expected_hours, ev.expected_fuel,
            ev.p_breach, ev.p_breach_upper, ev.beyond_horizon_fraction, list(chosen.labels), plan,
        ))

    rule = SELECTION_RULE.format(tol=fuel_tolerance)
    best = select_departure(options, fuel_tolerance)
    if best is None:
        return DepartureSweep(options, None, rule,
                              f"No departure date in the window satisfies the {risk_budget:.1%} risk budget.")
    feasible = [o for o in options if o.feasible]
    return DepartureSweep(
        options, best, rule,
        (f"{len(feasible)} of {len(options)} departure dates meet the {risk_budget:.1%} risk budget; "
         f"selected {best.departure.isoformat()} (E[fuel index]={best.expected_fuel:.0f}, "
         f"E[time]={best.expected_hours:.1f} h, P(breach) upper bound {best.p_breach_upper:.1%})."),
    )
