"""Static figures: route map on the polar grid and departure-window chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from antarctic_routing import DISCLAIMER  # noqa: E402
from antarctic_routing.routing.candidates import PlanResult  # noqa: E402
from antarctic_routing.routing.departure import DepartureSweep  # noqa: E402
from antarctic_routing.routing.hazard import exceedance_probability  # noqa: E402
from antarctic_routing.synthetic import ScenarioSet  # noqa: E402

_ROUTE_COLOURS = ["#5b6b7f", "#8a6fb3", "#c27c2c", "#3a8f6b", "#b04a5a", "#4f86c6", "#9a9a3a"]


def _rotator(grid):
    """Display rotation turning the domain's central meridian to point up.

    EPSG:3031 puts Greenwich at +y, so for this sector north would point to the
    upper left. Analysis stays in EPSG:3031; only the figure is rotated.
    """
    lam = np.deg2rad(float(np.median(grid.lon2d)))
    c, s = np.cos(lam), np.sin(lam)

    def rot(x, y):
        x, y = np.asarray(x, float) / 1e3, np.asarray(y, float) / 1e3
        return x * c - y * s, x * s + y * c

    return rot


def _graticule(ax, grid, rot) -> None:
    from antarctic_routing.preprocessing.grid import PolarGrid

    lats = np.arange(np.floor(grid.lat2d.min()), np.ceil(grid.lat2d.max()) + 1, 2)
    lons = np.arange(np.floor(grid.lon2d.min() / 5) * 5, np.ceil(grid.lon2d.max() / 5) * 5 + 1, 5)
    for lat in lats:
        lo = np.linspace(lons.min(), lons.max(), 200)
        x, y = rot(*PolarGrid.to_xy(np.full_like(lo, lat), lo))
        ax.plot(x, y, color="#9aa7b5", lw=0.4, alpha=0.6, zorder=1)
        ax.annotate(f"{abs(lat):.0f}\u00b0S", (x[0], y[0]), fontsize=6, color="#7a8796")
    for lon in lons:
        la = np.linspace(lats.min(), lats.max(), 200)
        x, y = rot(*PolarGrid.to_xy(la, np.full_like(la, lon)))
        ax.plot(x, y, color="#9aa7b5", lw=0.4, alpha=0.6, zorder=1)
        ax.annotate(f"{abs(lon):.0f}\u00b0W", (x[-1], y[-1]), fontsize=6, color="#7a8796")


def plot_plan(world: ScenarioSet, plan: PlanResult, tau: float, path: str | Path, title: str) -> Path:
    grid = world.grid
    rot = _rotator(grid)
    half = grid.resolution_m / 2
    xe = np.r_[grid.x - half, grid.x[-1] + half]
    ye = np.r_[grid.y - half, grid.y[-1] + half]
    XE, YE = rot(*np.meshgrid(xe, ye))
    hours = plan.recommended.evaluation.expected_hours if plan.recommended else 24.0
    t_idx = world.time_index(hours / 2 if np.isfinite(hours) else 0.0)
    p_ice = exceedance_probability(world.conc[:, t_idx], tau)

    fig, ax = plt.subplots(figsize=(9, 8.6), dpi=130)
    im = ax.pcolormesh(XE, YE, np.ma.masked_invalid(p_ice), cmap="Blues", vmin=0, vmax=1, zorder=0)
    ax.pcolormesh(XE, YE, np.ma.masked_where(~world.land, world.land.astype(float)),
                  cmap=ListedColormap(["#d9d2c3"]), zorder=2)
    _graticule(ax, grid, rot)
    if world.berg is not None:
        XC, YC = rot(*np.meshgrid(grid.x, grid.y))
        p_berg = world.berg[:, t_idx].mean(axis=0)
        if p_berg.max() > 0:
            ax.contourf(XC, YC, p_berg, levels=[0.05, 0.25, 0.5, 1.01], colors=["#c9b3e6", "#9b72cf", "#5e3c99"],
                        alpha=0.75, zorder=2.5)
            ax.contour(XC, YC, p_berg, levels=[0.05], colors=["#5e3c99"], linewidths=0.8, zorder=2.6)
            ax.plot([], [], color="#9b72cf", lw=6, label="iceberg presence P >= 5/25/50%")

    def xy(cells):
        r = np.array([c[0] for c in cells])
        c = np.array([c[1] for c in cells])
        return rot(grid.x[c], grid.y[r])

    for i, cand in enumerate(plan.candidates):
        if cand is plan.recommended:
            continue
        x, y = xy(cand.route.cells)
        ax.plot(x, y, color=_ROUTE_COLOURS[i % len(_ROUTE_COLOURS)], lw=1.2, alpha=0.85, zorder=3,
                label=f"{cand.labels[0]} - P(breach) {cand.evaluation.p_breach:.0%}")
    if plan.recommended:
        x, y = xy(plan.recommended.route.cells)
        ev = plan.recommended.evaluation
        ax.plot(x, y, color="#d1495b", lw=3.0, zorder=4,
                label=f"RECOMMENDED - {ev.expected_hours:.0f} h, fuel {ev.expected_fuel:.0f}, "
                      f"P(breach) {ev.p_breach:.1%} (UB {ev.p_breach_upper:.1%})")
    first = plan.candidates[0].route.cells if plan.candidates else None
    if first:
        ox, oy = xy([first[0]])
        dx, dy = xy([first[-1]])
        ax.scatter(ox, oy, marker="o", s=70, color="#1b998b", edgecolor="white", zorder=5, label="Origin")
        ax.scatter(dx, dy, marker="*", s=180, color="#f4a259", edgecolor="black", zorder=5, label="Destination")

    ax.set_xlim(XE.min(), XE.max())
    ax.set_ylim(YE.min(), YE.max())
    ax.set_aspect("equal")
    ax.set_xlabel("km (EPSG:3031, display rotated: local north up)")
    ax.set_ylabel("km")
    ax.set_title(f"{title}\nstatus: {plan.status.upper()}  |  background: P(C >= {tau:.2f}) at day {t_idx}",
                 fontsize=10)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="P(ice concentration >= vessel limit)")
    fig.text(0.01, 0.005, f"{world.execution_mode.upper()} DATA - {DISCLAIMER}", fontsize=6, color="#555", wrap=True)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_departures(sweep: DepartureSweep, risk_budget: float, path: str | Path, title: str,
                    issue=None, forecast_days: int | None = None, trust_horizon_days: int | None = None) -> Path:
    opts = sweep.options
    dates = [o.departure for o in opts]
    ub = [min(o.p_breach_upper, 1.0) for o in opts]
    hours = [o.expected_hours if np.isfinite(o.expected_hours) else np.nan for o in opts]
    fuel = [o.expected_fuel if np.isfinite(o.expected_fuel) else np.nan for o in opts]
    ok = [o.feasible for o in opts]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), dpi=130, sharex=True)
    bars = axes[0].bar(dates, ub, color=["#3a8f6b" if f else "#d1495b" for f in ok], width=0.8)
    for bar, opt in zip(bars, opts, strict=True):
        if opt.support == "climatology-dominated":
            bar.set_hatch("///")
            bar.set_edgecolor("white")
    if issue is not None:
        from datetime import timedelta

        marks = []
        if forecast_days is not None:
            marks.append((issue + timedelta(days=forecast_days), "#4f86c6", f"forecast horizon ({forecast_days} d)"))
        if trust_horizon_days is not None:
            marks.append((issue + timedelta(days=trust_horizon_days), "#7a4fb3",
                          f"trust horizon ({trust_horizon_days} d)"))
        for when, colour, label in marks:
            for ax in axes:
                ax.axvline(when, color=colour, ls="--", lw=1.2)
            axes[0].text(when, 1.02, " " + label, color=colour, fontsize=7, va="bottom")
        axes[0].bar([], [], color="#888", hatch="///", edgecolor="white", label="climatology-dominated voyage")
    axes[0].axhline(risk_budget, color="black", ls="--", lw=1, label=f"risk budget {risk_budget:.0%}")
    axes[0].set_ylabel("P(breach)\nupper bound")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    axes[1].plot(dates, hours, marker="o", color="#4f86c6")
    axes[1].set_ylabel("E[voyage] (h)")
    axes[2].plot(dates, fuel, marker="o", color="#c27c2c")
    axes[2].set_ylabel("E[fuel index]")
    if sweep.selected:
        for ax in axes:
            ax.axvline(sweep.selected.departure, color="#1b998b", lw=2, alpha=0.6)
        axes[0].text(sweep.selected.departure, 0.95, " selected", color="#1b998b", fontsize=8, va="top")
    axes[0].set_title(f"{title}\n{sweep.rule}", fontsize=9)
    fig.autofmt_xdate()
    fig.text(0.01, 0.005, DISCLAIMER, fontsize=6, color="#555", wrap=True)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


_METHOD_STYLE = {
    "persistence": ("#8a8f98", "--"),
    "climatology": ("#c27c2c", ":"),
    "damped_persistence": ("#4f86c6", "-."),
}


def plot_forecast_skill(report: dict, path: str | Path, title: str) -> Path:
    """MAE and IIEE against lead time for the model and every baseline."""
    leads = report["leads"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=130)
    for method in report["methods"]:
        colour, ls = _METHOD_STYLE.get(method, ("#d1495b", "-"))
        lw = 2.6 if method not in _METHOD_STYLE else 1.5
        axes[0].plot(leads, report["mae"][method], ls, color=colour, lw=lw, marker="o", ms=3, label=method)
        axes[1].plot(leads, np.asarray(report["iiee_km2"][method]) / 1e3, ls, color=colour, lw=lw,
                     marker="o", ms=3, label=method)
    axes[0].set_ylabel("MAE (concentration fraction)")
    axes[1].set_ylabel(f"IIEE (10^3 km^2, edge at C >= {report['edge_threshold']})")
    for ax in axes:
        ax.set_xlabel("lead time (days)")
        ax.grid(alpha=0.3)
        ax.set_xticks(leads)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{title}\n{report['execution_mode'].upper()} - test seasons {report['test_seasons']}, "
                 f"{report['n_samples']} forecasts, rho={report['rho']:.3f}", fontsize=10)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_reliability(report: dict, path: str | Path, title: str) -> Path:
    """Reliability diagram (raw vs calibrated vs climatology) and Brier score per lead."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=130)
    styles = {"raw": ("#8a8f98", "o--"), "calibrated": ("#d1495b", "o-"), "climatology": ("#c27c2c", "s:")}
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="black", lw=0.8, alpha=0.6, label="perfect reliability")
    for name, (colour, fmt) in styles.items():
        rows = [r for r in report["reliability"][name] if r["count"]]
        ax.plot([r["mean_predicted"] for r in rows], [r["observed_frequency"] for r in rows], fmt,
                color=colour, ms=4, label=name)
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_title(f"Reliability, P(C >= {report['tau']}) pooled over leads", fontsize=9)
    ax = axes[1]
    for name, (colour, fmt) in styles.items():
        ax.plot(report["leads"], report["brier"][name], fmt, color=colour, ms=4, label=name)
    ax.set_xlabel("lead time (days)")
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_xticks(report["leads"])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle(f"{title}\n{report['execution_mode'].upper()} - calibrated on {report['val_seasons']}, "
                 f"tested on {report['test_seasons']}, {report['n_members']} members", fontsize=10)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_trust_horizon(result: dict, path: str | Path, title: str) -> Path:
    """Delta(h) = E_baseline - E_model with bootstrap bands and the trust-horizon marker."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=130)
    colours = {"damped_persistence": "#4f86c6", "climatology": "#c27c2c", "persistence": "#8a8f98"}
    for name, th in result["by_baseline"].items():
        c = colours.get(name, "#555")
        ax.fill_between(th["leads"], th["ci_low"], th["ci_high"], color=c, alpha=0.18)
        ax.plot(th["leads"], th["delta"], "o-", color=c, ms=4,
                label=f"vs {name}: trust horizon {th['trust_horizon_days']} d")
    ax.axhline(result["min_delta"], color="black", lw=0.8, ls="--", label="required improvement")
    h = result["trust_horizon_days"]
    if h:
        ax.axvspan(0.5, h + 0.5, color="#3a8f6b", alpha=0.08)
        ax.text(h + 0.45, ax.get_ylim()[1] * 0.92, f"trusted up to {h} d", ha="right", color="#3a8f6b", fontsize=9)
    ax.set_xlabel("lead time (days)")
    ax.set_ylabel("MAE improvement over baseline")
    ax.set_xticks(result["leads"])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title(f"{title}\n{result['execution_mode'].upper()} - {result['method']}, "
                 f"{result['n_seasons']} seasons, {int(100 * (1 - result['alpha']))}% lower bound", fontsize=9)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
