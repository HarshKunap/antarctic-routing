"""Trust horizon (Stage 8.4)."""

import numpy as np
import pytest

from antarctic_routing.forecasting.trust import trust_horizon

SEASONS = [2018, 2019, 2020, 2021, 2022, 2023]


def _by_season(model_errors, baseline_errors, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = {"model": {}, "base": {}}
    for s in SEASONS:
        out["model"][s] = (np.array(model_errors) + rng.normal(0, jitter, len(model_errors))).tolist()
        out["base"][s] = list(baseline_errors)
    return out


def test_horizon_is_last_lead_with_supported_improvement():
    bs = _by_season([0.01, 0.02, 0.03, 0.05, 0.05], [0.02, 0.03, 0.04, 0.04, 0.06], jitter=0.001)
    th = trust_horizon(bs, "model", "base", n_boot=500, seed=0)
    assert th["trust_horizon_days"] == 3          # lead 4 is worse, so lead 5 cannot extend it
    assert th["delta"][0] == pytest.approx(0.01, abs=0.002)
    assert all(lo <= d <= hi for lo, d, hi in zip(th["ci_low"], th["delta"], th["ci_high"], strict=True))


def test_horizon_is_flagged_when_limited_by_evaluated_leads():
    bs = _by_season([0.01, 0.02, 0.03], [0.02, 0.04, 0.06], jitter=0.001)
    th = trust_horizon(bs, "model", "base", n_boot=300)
    assert th["trust_horizon_days"] == 3 and th["limited_by_max_lead"] is True


def test_no_trust_when_model_loses_at_lead_one():
    bs = _by_season([0.03, 0.01], [0.02, 0.05])
    assert trust_horizon(bs, "model", "base", n_boot=200)["trust_horizon_days"] == 0


def test_inconsistent_gains_across_seasons_are_not_trusted():
    bs = {"model": {}, "base": {}}
    for i, s in enumerate(SEASONS):
        sign = 1 if i % 2 else -1
        bs["model"][s] = [0.02 - sign * 0.01]
        bs["base"][s] = [0.02]
    th = trust_horizon(bs, "model", "base", n_boot=500)
    assert th["ci_low"][0] < 0 < th["ci_high"][0]
    assert th["trust_horizon_days"] == 0


def test_min_delta_requires_operationally_meaningful_gain():
    bs = _by_season([0.0195, 0.019], [0.02, 0.02])
    assert trust_horizon(bs, "model", "base", n_boot=200, min_delta=0.0)["trust_horizon_days"] == 2
    assert trust_horizon(bs, "model", "base", n_boot=200, min_delta=0.002)["trust_horizon_days"] == 0


def test_few_seasons_produce_a_warning_and_results_are_seeded():
    bs = {"model": {2020: [0.01], 2021: [0.012]}, "base": {2020: [0.02], 2021: [0.02]}}
    a = trust_horizon(bs, "model", "base", n_boot=300, seed=3)
    b = trust_horizon(bs, "model", "base", n_boot=300, seed=3)
    assert a == b
    assert any("seasons" in w for w in a["warnings"])


def test_evaluation_report_contains_per_season_errors():
    from pathlib import Path

    from antarctic_routing.config import load_config
    from antarctic_routing.forecasting.evaluate import evaluate_forecasts
    from antarctic_routing.preprocessing.grid import PolarGrid
    from antarctic_routing.synthetic import synthetic_history

    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    ds = synthetic_history(cfg, PolarGrid.from_domain(cfg.domain, 50), range(2010, 2016), seed=1)
    conc = np.nan_to_num(ds["ice_concentration"].values)
    report = evaluate_forecasts(ds, lambda x, idx: np.stack([np.repeat(conc[t][None], 2, 0) for t in idx]),
                                [2010, 2011, 2012], [2014, 2015], 5, 2, cfg.project.season_months)
    assert set(report["by_season"]["unet"]) == {2014, 2015}
    assert len(report["by_season"]["damped_persistence"][2015]) == 2
    pooled = np.mean([report["by_season"]["persistence"][s][0] for s in (2014, 2015)])
    assert pooled == pytest.approx(report["mae"]["persistence"][0], rel=0.05)
