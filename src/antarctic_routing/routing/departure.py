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

Two ways to build the window:

* :func:`sweep_departures` - fresh scenarios per date (e.g. replaying what was
  known on each date);
* :func:`plan_from_issue` - ONE forecast issued today; departing j days later
  sails through layers j, j+1, ... of the same joint scenarios. Each option
  reports how much of the voyage is forecast-supported versus
  climatology-dominated, and whether it stays inside the trust horizon.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

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
    lead_days: int | None = None
    forecast_fraction: float | None = None
    support: str | None = None
    within_trust_horizon: bool | None = None

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
            "lead_days": self.lead_days,
            "forecast_fraction": self.forecast_fraction,
            "support": self.support,
            "within_trust_horizon": self.within_trust_horizon,
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


def _option(d: date, plan: PlanResult) -> DepartureOption:
    chosen = plan.recommended or next((c for c in plan.candidates if "lowest_risk" in c.tags), None)
    if chosen is None:
        return DepartureOption(d, plan.status, False, float("inf"), float("inf"), 1.0, 1.0, 0.0, [], plan)
    ev = chosen.evaluation
    return DepartureOption(d, plan.status, plan.status == "feasible", ev.expected_hours, ev.expected_fuel,
                           ev.p_breach, ev.p_breach_upper, ev.beyond_horizon_fraction, list(chosen.labels), plan)


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
        options.append(_option(d, plan))

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


def _forecast_fraction(world: ScenarioSet, start_h: float, hours: float) -> float | None:
    """Share of the voyage [start_h, start_h + hours] spent in observed/forecast layers."""
    if world.layer_source is None:
        return None
    if not hours or hours != hours or hours == float("inf"):
        return 0.0
    step, end, covered = world.time_step_hours, start_h + hours, 0.0
    for i, src in enumerate(world.layer_source):
        lo, hi = i * step, (i + 1) * step
        if src != "climatology":
            covered += max(0.0, min(hi, end) - max(lo, start_h))
    if end > world.horizon_hours:  # time past the last layer reuses it; count it as unsupported
        covered = min(covered, world.horizon_hours - start_h)
    return round(covered / hours, 6)


def plan_from_issue(
    world: ScenarioSet,
    offsets_days: Sequence[int],
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
    trust_horizon_days: int | None = None,
    require_trusted: bool = False,
    fuel_tolerance: float = 0.01,
) -> DepartureSweep:
    """Departure window from one forecast issue; option j departs ``j`` days after issue."""
    step = world.time_step_hours
    for j in offsets_days:
        if j < 0 or j * step >= world.horizon_hours:
            raise ValueError(f"departure offset {j} d is outside the {world.horizon_hours / 24:g} d scenario horizon")
    rgrid = RoutingGrid.build(world.grid, world.ocean, connectivity)
    issue = world.start.date()
    options: list[DepartureOption] = []
    for j in offsets_days:
        start_h = j * step
        plan = plan_candidates(
            world, vessel, origin, destination, risk_budget, risk_weights, estimator, connectivity,
            n_scenario_routes, confidence, seed, depart_hours=start_h, rgrid=rgrid,
        )
        opt = _option(issue + timedelta(days=j), plan)
        opt.lead_days = j
        opt.forecast_fraction = _forecast_fraction(world, start_h, opt.expected_hours)
        if opt.forecast_fraction is not None:
            opt.support = "forecast-supported" if opt.forecast_fraction >= 0.5 else "climatology-dominated"
        if trust_horizon_days is not None and opt.expected_hours != float("inf"):
            last_layer = int((start_h + opt.expected_hours) // step)
            opt.within_trust_horizon = last_layer <= trust_horizon_days
        options.append(opt)

    rule = SELECTION_RULE.format(tol=fuel_tolerance)
    if require_trusted:
        rule += "; only dates whose whole voyage lies within the trust horizon are eligible"
    eligible = [o for o in options if o.feasible and (not require_trusted or o.within_trust_horizon)]
    best = select_departure(eligible, fuel_tolerance)
    if best is None:
        why = " within the trust horizon" if require_trusted else ""
        return DepartureSweep(options, None, rule,
                              f"No departure date{why} satisfies the {risk_budget:.1%} risk budget "
                              f"(forecast issued {issue.isoformat()}).")
    return DepartureSweep(
        options, best, rule,
        (f"Forecast issued {issue.isoformat()}: {len(eligible)} of {len(options)} departure dates eligible; "
         f"selected {best.departure.isoformat()} (+{best.lead_days} d, {best.support or 'n/a'}, "
         f"E[fuel index]={best.expected_fuel:.0f}, E[time]={best.expected_hours:.1f} h, "
         f"P(breach) upper bound {best.p_breach_upper:.1%})."),
    )
