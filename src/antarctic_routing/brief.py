"""Stage 11.3 - PDF voyage brief.

Page 1 summarises the decision (departure, recommended route, expected time,
fuel index, risk-budget status, data mode, key assumptions, limitations);
page 2 is the route map; page 3 compares every candidate evaluated on the same
joint scenarios.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from antarctic_routing import DISCLAIMER, __version__  # noqa: E402
from antarctic_routing.common.provenance import utc_now  # noqa: E402
from antarctic_routing.config import ProjectConfig  # noqa: E402
from antarctic_routing.routing.candidates import PlanResult  # noqa: E402
from antarctic_routing.synthetic import ScenarioSet  # noqa: E402

_A4 = (8.27, 11.69)


def _text_page(pdf: PdfPages, title: str, lines: list[tuple[str, str]], notes: list[str]) -> None:
    fig = plt.figure(figsize=_A4)
    fig.text(0.08, 0.95, title, fontsize=16, weight="bold")
    fig.text(0.08, 0.925, f"Generated {utc_now()} - antarctic-routing {__version__}", fontsize=8, color="#555")
    y = 0.89
    for key, value in lines:
        wrapped = textwrap.fill(str(value), width=62)
        fig.text(0.08, y, key, fontsize=10, color="#52514e", va="top")
        fig.text(0.38, y, wrapped, fontsize=10, va="top")
        y -= 0.026 * (wrapped.count("\n") + 1) + 0.01
    y -= 0.02
    fig.text(0.08, y, "Assumptions and limitations", fontsize=11, weight="bold", va="top")
    y -= 0.03
    for note in notes:
        wrapped = textwrap.fill("- " + note, width=105, subsequent_indent="  ")
        fig.text(0.09, y, wrapped, fontsize=8.5, va="top")
        y -= 0.02 * (wrapped.count("\n") + 1) + 0.008
    fig.text(0.08, 0.06, textwrap.fill(DISCLAIMER, width=110), fontsize=8, color="#d03b3b", va="top")
    pdf.savefig(fig)
    plt.close(fig)


def write_brief(
    path: str | Path,
    cfg: ProjectConfig,
    world: ScenarioSet,
    plan: PlanResult,
    map_png: str | Path,
    trust_horizon_days: int | None = None,
    trust_limited: bool = False,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = plan.recommended
    ev = rec.evaluation if rec else None
    lines = [
        ("Route", f"({cfg.route.origin.lat}, {cfg.route.origin.lon}) -> "
                  f"({cfg.route.destination.lat}, {cfg.route.destination.lon})"),
        ("Departure (UTC)", world.start.isoformat()),
        ("Decision", "RECOMMENDED ROUTE" if rec else "NO ROUTE MEETS THE RISK BUDGET"),
        ("Risk budget", f"{cfg.routing.risk_budget:.1%} via {cfg.routing.risk_estimator} at "
                        f"{cfg.routing.confidence:.0%} confidence"),
        ("Joint scenarios", str(world.n_scenarios)),
        ("Data mode", world.execution_mode.upper()),
    ]
    if ev:
        lines += [
            ("Route labels", ", ".join(rec.labels)),
            ("P(breach)", f"{ev.p_breach:.1%} (95% upper bound {ev.p_breach_upper:.1%})"),
            ("Expected time", f"{ev.expected_hours:.1f} h (P10-P90 {ev.hours_p10:.1f}-{ev.hours_p90:.1f} h)"),
            ("Fuel index", f"{ev.expected_fuel:.0f} (relative, not litres)"),
            ("Distance", f"{ev.distance_km:.0f} km"),
            ("Beyond scenario horizon", f"{ev.beyond_horizon_fraction:.0%} of scenarios"),
        ]
    if trust_horizon_days is not None:
        lines.append(("Forecast trust horizon", f">= {trust_horizon_days} days (limited by evaluated leads)"
                      if trust_limited else f"{trust_horizon_days} days"))
    notes = [
        f"Vessel ice limit tau_v = {cfg.vessel.max_ice_concentration} and ice class "
        f"'{cfg.vessel.ice_class}' are placeholders; derive them from verified vessel capability.",
        "Fuel is a relative index d[1 + lambda C^2]; speed in ice v(1 - kC) is an assumption (see sensitivity).",
        "Route risk is evaluated jointly across scenarios; it is a model estimate, not a guarantee.",
        "Tracked icebergs (USNIC) cover giant bergs only; smaller bergs and growlers are not represented.",
        "Geography in controlled-synthetic runs is schematic and must not be used for navigation.",
        plan.explanation,
    ]
    with PdfPages(out) as pdf:
        _text_page(pdf, "Voyage brief - Antarctic ice-risk routing", lines, notes)
        fig = plt.figure(figsize=_A4)
        img = plt.imread(str(map_png))
        ax = fig.add_axes([0.04, 0.1, 0.92, 0.82])
        ax.imshow(img)
        ax.axis("off")
        fig.text(0.08, 0.95, "Route map", fontsize=14, weight="bold")
        pdf.savefig(fig)
        plt.close(fig)
        fig = plt.figure(figsize=_A4)
        fig.text(0.08, 0.95, "Candidate routes (same joint scenarios)", fontsize=14, weight="bold")
        ax = fig.add_axes([0.05, 0.3, 0.9, 0.6])
        ax.axis("off")
        rows = [[", ".join(c.labels)[:38], "yes" if c.feasible else "no", f"{c.evaluation.distance_km:.0f}",
                 f"{c.evaluation.expected_hours:.1f}", f"{c.evaluation.expected_fuel:.0f}",
                 f"{c.evaluation.p_breach:.1%}", f"{c.evaluation.p_breach_upper:.1%}"] for c in plan.candidates]
        table = ax.table(cellText=rows, colLabels=["route", "meets budget", "km", "E[h]", "E[fuel]", "P(breach)",
                                                   "95% UB"], loc="upper center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)
        pdf.savefig(fig)
        plt.close(fig)
        info = pdf.infodict()
        info["Title"] = "Antarctic voyage brief"
        info["Subject"] = DISCLAIMER
    return out
