from datetime import date
from pathlib import Path

import numpy as np
import pytest
import torch

from antarctic_routing.config import load_config
from antarctic_routing.forecasting.dataset import SequenceDataset, build_samples
from antarctic_routing.forecasting.evaluate import evaluate_forecasts
from antarctic_routing.forecasting.train import TrainConfig, load_model, masked_mae_loss, train_unet
from antarctic_routing.forecasting.unet import IceUNet
from antarctic_routing.preprocessing.climatology import season_of
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.synthetic import synthetic_history

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
SEASON = CFG.project.season_months


@pytest.fixture(scope="module")
def grid():
    return PolarGrid.from_domain(CFG.domain, resolution_km=50)


@pytest.fixture(scope="module")
def history(grid):
    return synthetic_history(CFG, grid, seasons=range(2010, 2016), seed=3)


# ------------------------------------------------------------ synthetic history
def test_history_covers_only_season_days_and_is_labelled(history):
    days = [np.datetime64(t, "D").astype(object) for t in history["time"].values]
    assert {season_of(d, SEASON) for d in days} == set(range(2010, 2016))
    assert history.attrs["execution_mode"] == "controlled_synthetic"
    conc = history["ice_concentration"].values[:, ~history["land_mask"].values]
    assert np.isfinite(conc).all() and conc.min() >= 0 and conc.max() <= 1


def test_history_is_deterministic(grid):
    a = synthetic_history(CFG, grid, seasons=[2010], seed=1)["ice_concentration"].values
    b = synthetic_history(CFG, grid, seasons=[2010], seed=1)["ice_concentration"].values
    assert np.array_equal(a, b, equal_nan=True)


def test_history_has_dynamics_persistence_error_grows_with_lead(history):
    c = history["ice_concentration"].values[:100]
    ocean = ~history["land_mask"].values
    err = [np.mean(np.abs(c[h:][:, ocean] - c[:-h][:, ocean])) for h in (1, 7)]
    assert err[1] > 2 * err[0]


# ------------------------------------------------------------------- samples
def test_samples_never_cross_season_boundaries(history):
    idx = build_samples(history, history_days=5, lead_days=3, season_months=SEASON)
    times = history["time"].values
    for t in idx:
        window = [np.datetime64(x, "D").astype(object) for x in times[t - 4: t + 4]]
        assert len({season_of(d, SEASON) for d in window}) == 1
        assert (np.diff(times[t - 4: t + 4]) == np.timedelta64(1, "D")).all()
    per_season = 120 - 5 - 3 + 1
    assert len(idx) == 6 * per_season


def test_sequence_dataset_aligns_inputs_and_targets(history):
    idx = build_samples(history, 5, 3, SEASON)
    ds = SequenceDataset(history, idx, history_days=5, lead_days=3, season_months=SEASON)
    x, y, mask = ds[0]
    t = idx[0]
    conc = np.nan_to_num(history["ice_concentration"].values, nan=0.0)
    assert x.shape[0] == 5 + 3  # 5 history frames + land + sin + cos
    assert np.allclose(x[4].numpy(), conc[t])          # last input frame is "today"
    assert np.allclose(y[2].numpy(), conc[t + 3])      # target channel h-1 is t + h
    assert mask.dtype == torch.bool and not mask[history["land_mask"].values].any()


# --------------------------------------------------------------------- model
@pytest.mark.parametrize("shape", [(23, 31), (32, 32), (17, 9)])
def test_unet_preserves_spatial_shape_and_outputs_fractions(shape):
    torch.manual_seed(0)
    model = IceUNet(in_channels=8, out_channels=3, base=8)
    out = model(torch.rand(2, 8, *shape))
    assert out.shape == (2, 3, *shape)
    assert out.min() >= 0 and out.max() <= 1


def test_masked_loss_ignores_land():
    pred = torch.zeros(1, 1, 2, 2)
    target = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    mask = torch.tensor([[False, True], [True, True]])
    assert masked_mae_loss(pred, target, mask).item() == pytest.approx(0.0)


# ---------------------------------------------------------- train & evaluate
def test_training_reduces_validation_error_and_checkpoint_roundtrips(history, tmp_path):
    tc = TrainConfig(history_days=5, lead_days=3, epochs=4, batch_size=16, base_channels=8, lr=3e-3, seed=0)
    result = train_unet(history, train_seasons=[2010, 2011, 2012, 2013], val_seasons=[2014],
                        season_months=SEASON, cfg=tc, out_dir=tmp_path)
    assert result.history[-1]["val_mae"] < result.history[0]["val_mae"]
    model, meta = load_model(tmp_path / "best.pt")
    assert meta["train_seasons"] == [2010, 2011, 2012, 2013]
    assert meta["execution_mode"] == "controlled_synthetic"
    x = torch.rand(1, 8, *history["land_mask"].shape)
    with torch.no_grad():
        assert torch.allclose(model(x), result.model.eval()(x))


def test_evaluation_reports_every_method_per_lead_and_oracle_is_perfect(history):
    conc = np.nan_to_num(history["ice_concentration"].values, nan=0.0)

    def oracle(batch_inputs, sample_index):
        return np.stack([conc[t + 1: t + 4] for t in sample_index])

    report = evaluate_forecasts(history, oracle, train_seasons=[2010, 2011, 2012, 2013], test_seasons=[2015],
                                history_days=5, lead_days=3, season_months=SEASON, model_name="oracle")
    assert set(report["methods"]) == {"oracle", "persistence", "climatology", "damped_persistence"}
    for method in report["methods"]:
        assert len(report["mae"][method]) == 3
    assert max(report["mae"]["oracle"]) < 1e-6
    assert report["mae"]["persistence"][0] < report["mae"]["persistence"][2]
    assert 0.0 <= report["rho"] <= 1.0
    assert report["iiee_km2"]["oracle"][0] == 0.0
    assert report["n_samples"] == 120 - 5 - 3 + 1
    assert report["test_seasons"] == [2015]


def test_evaluation_rejects_overlapping_train_and_test_seasons(history):
    with pytest.raises(ValueError, match="overlap"):
        evaluate_forecasts(history, lambda b, i: None, [2010, 2015], [2015], 5, 3, SEASON)


def test_issue_date_helper():
    assert season_of(date(2027, 1, 5), SEASON) == 2026


def test_residual_unet_with_zero_head_reproduces_persistence_exactly():
    torch.manual_seed(0)
    model = IceUNet(in_channels=8, out_channels=3, base=8, persistence_channel=4).eval()
    x = torch.rand(2, 8, 19, 23)
    with torch.no_grad():
        out = model(x)
    assert torch.allclose(out, x[:, 4:5].expand_as(out))   # zero-initialised head => C_t for every lead


def test_residual_unet_output_stays_in_unit_interval():
    torch.manual_seed(1)
    model = IceUNet(in_channels=8, out_channels=3, base=8, persistence_channel=4)
    with torch.no_grad():
        model.head.weight.normal_(0, 5.0)
        model.head.bias.normal_(0, 5.0)
    out = model(torch.rand(2, 8, 16, 16))
    assert out.min() >= 0 and out.max() <= 1
