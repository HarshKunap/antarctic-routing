"""Train the sea-ice U-Net with a masked MAE loss and early stopping.

    L_MAE = sum_i M_i |C_hat_i - C_i| / sum_i M_i      (M_i = 1 on ocean cells)

The best epoch (lowest validation MAE) is checkpointed together with the
metadata needed to reproduce and audit it: seasons used, window lengths, seed
and the execution mode of the training data.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

from antarctic_routing.common.provenance import utc_now
from antarctic_routing.forecasting.dataset import SequenceDataset, build_samples, issue_seasons
from antarctic_routing.forecasting.unet import IceUNet


@dataclass
class TrainConfig:
    history_days: int = 14
    lead_days: int = 7
    epochs: int = 30
    batch_size: int = 16
    base_channels: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 6
    seed: int = 0
    residual: bool = True


@dataclass
class TrainResult:
    model: IceUNet
    history: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    checkpoint: Path | None = None


def masked_mae_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over cells where ``mask`` is True.

    ``mask`` may be (H, W), (B, H, W) or (B, 1, H, W); it is broadcast over channels.
    """
    m = mask.to(pred.dtype)
    if m.dim() == 2:
        m = m[None, None]
    elif m.dim() == 3:
        m = m[:, None]
    m = m.expand_as(pred)
    return (torch.abs(pred - target) * m).sum() / m.sum().clamp_min(1.0)


def _subset(ds: xr.Dataset, cfg: TrainConfig, season_months, seasons: Sequence[int]) -> SequenceDataset:
    idx = build_samples(ds, cfg.history_days, cfg.lead_days, season_months)
    wanted = set(seasons)
    idx = [t for t, s in zip(idx, issue_seasons(ds, idx, season_months), strict=True) if s in wanted]
    if not idx:
        raise ValueError(f"no samples for seasons {sorted(wanted)}")
    return SequenceDataset(ds, idx, cfg.history_days, cfg.lead_days, season_months)


def _evaluate(model: IceUNet, loader: DataLoader) -> float:
    model.eval()
    total, weight = 0.0, 0.0
    with torch.no_grad():
        for x, y, mask in loader:
            m = mask[:, None].expand_as(y).float()
            total += float((torch.abs(model(x) - y) * m).sum())
            weight += float(m.sum())
    return total / max(weight, 1.0)


def train_unet(
    ds: xr.Dataset,
    train_seasons: Sequence[int],
    val_seasons: Sequence[int],
    season_months: Sequence[int],
    cfg: TrainConfig,
    out_dir: str | Path,
) -> TrainResult:
    if set(train_seasons) & set(val_seasons):
        raise ValueError("train and validation seasons overlap")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    train = _subset(ds, cfg, season_months, train_seasons)
    val = _subset(ds, cfg, season_months, val_seasons)
    gen = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val, batch_size=cfg.batch_size)

    model = IceUNet(train.in_channels, cfg.lead_days, base=cfg.base_channels,
                    persistence_channel=cfg.history_days - 1 if cfg.residual else None)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = TrainResult(model=model)
    best, best_state, stale = float("inf"), None, 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, batches = 0.0, 0
        for x, y, mask in train_loader:
            opt.zero_grad()
            loss = masked_mae_loss(model(x), y, mask[:, None])
            loss.backward()
            opt.step()
            running += loss.item()
            batches += 1
        sched.step()
        val_mae = _evaluate(model, val_loader)
        result.history.append({"epoch": epoch, "train_loss": running / max(batches, 1), "val_mae": val_mae})
        if val_mae < best - 1e-6:
            best, stale, result.best_epoch = val_mae, 0, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    meta = {
        "in_channels": train.in_channels, "out_channels": cfg.lead_days, "base_channels": cfg.base_channels,
        "persistence_channel": model.persistence_channel,
        "train_config": asdict(cfg), "train_seasons": sorted(train_seasons), "val_seasons": sorted(val_seasons),
        "season_months": list(season_months), "best_epoch": result.best_epoch, "val_mae": best,
        "execution_mode": ds.attrs.get("execution_mode", "real"), "grid_shape": list(train.land.shape),
        "created_at": utc_now(),
    }
    result.checkpoint = out / "best.pt"
    torch.save({"state_dict": model.state_dict(), "meta": meta}, result.checkpoint)
    (out / "history.json").write_text(json.dumps({"history": result.history, "meta": meta}, indent=2))
    return result


def load_model(path: str | Path) -> tuple[IceUNet, dict]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    meta = blob["meta"]
    model = IceUNet(meta["in_channels"], meta["out_channels"], base=meta["base_channels"],
                    persistence_channel=meta.get("persistence_channel"))
    model.load_state_dict(blob["state_dict"])
    return model.eval(), meta


def unet_predictor(model: IceUNet, batch_size: int = 32):
    """Adapter giving a U-Net the ``predictor(inputs, indices)`` interface used by evaluation."""

    def predict(inputs: np.ndarray, indices: Sequence[int]) -> np.ndarray:
        outs = []
        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                outs.append(model(torch.from_numpy(np.asarray(inputs[i: i + batch_size]))).numpy())
        return np.concatenate(outs)

    return predict
