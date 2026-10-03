"""Stage 8.1-8.2 - joint scenarios from a forecast issued on a given day.

Layer layout for a forecast issued on day t (one layer per day):

=========  ==============  =====================================================
layer      source          content
=========  ==============  =====================================================
0          ``observed``    C_t, identical in every member
1 .. H     ``forecast``    model forecast + one whole training-season residual
                           sequence per member (coherent in space and lead)
> H        ``climatology`` clip(mu(d) + A^(k)(d)): member k replays the daily
                           anomaly *sequence* of one training season, so spatial
                           and temporal correlation are preserved
=========  ==============  =====================================================

Only information available on day t is used: the model sees the input window
ending at t, and the residual bank, climatology and anomaly library come from
training seasons. Repeating the final forecast beyond H is deliberately avoided.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import xarray as xr

from antarctic_routing.forecasting.calibration import residual_ensemble
from antarctic_routing.forecasting.dataset import SequenceDataset, build_samples, issue_seasons
from antarctic_routing.preprocessing.climatology import Climatology, day_of_season, season_of
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.synthetic import ScenarioSet, schematic_currents, schematic_winds

Predictor = Callable[[np.ndarray, Sequence[int]], np.ndarray]


def grid_of(ds: xr.Dataset) -> PolarGrid:
    return PolarGrid(x=ds["x"].values, y=ds["y"].values, resolution_m=float(ds.attrs["resolution_m"]))


@dataclass
class ForecastContext:
    ds: xr.Dataset
    predictor: Predictor
    history_days: int
    lead_days: int
    season_months: list[int]
    train_seasons: list[int]
    clim: Climatology
    bank: np.ndarray                     # (N, H, ny, nx) residual sequences
    bank_seasons: list[int]
    anomalies: dict[int, dict[int, np.ndarray]] = field(repr=False)  # season -> day-of-season -> field
    currents: tuple[np.ndarray, np.ndarray] | None = None
    winds: tuple[np.ndarray, np.ndarray] | None = None

    @classmethod
    def build(
        cls,
        ds: xr.Dataset,
        predictor: Predictor,
        history_days: int,
        lead_days: int,
        season_months: Sequence[int],
        train_seasons: Sequence[int],
        bank_size: int = 300,
        seed: int = 0,
        currents: tuple[np.ndarray, np.ndarray] | None = None,
        winds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> ForecastContext:
        months = list(season_months)
        days = [np.datetime64(t, "D").astype(object) for t in ds["time"].values]
        conc = ds["ice_concentration"].values.astype(float)
        clim = Climatology.fit(conc, days, train_seasons, months, window_days=7)

        idx = build_samples(ds, history_days, lead_days, months)
        train = set(train_seasons)
        idx = [t for t, s in zip(idx, issue_seasons(ds, idx, months), strict=True) if s in train]
        if not idx:
            raise ValueError("no training samples for the residual bank")
        rng = np.random.default_rng(seed)
        picks = sorted(rng.choice(idx, size=min(bank_size, len(idx)), replace=False).tolist())
        seq = SequenceDataset(ds, [], history_days, lead_days, months)
        preds = np.concatenate([
            np.asarray(predictor(np.stack([seq.inputs(t) for t in picks[i: i + 32]]), picks[i: i + 32]), float)
            for i in range(0, len(picks), 32)
        ])
        targets = np.stack([conc[t + 1: t + lead_days + 1] for t in picks])
        bank = np.nan_to_num(targets - preds, nan=0.0).astype(np.float32)

        anomalies: dict[int, dict[int, np.ndarray]] = {}
        for i, d in enumerate(days):
            s = season_of(d, months)
            if s in train:
                anomalies.setdefault(s, {})[day_of_season(d, months)] = np.nan_to_num(clim.anomaly(conc[i], d))

        grid = grid_of(ds)
        land = ds["land_mask"].values.astype(bool)
        return cls(
            ds=ds, predictor=predictor, history_days=history_days, lead_days=lead_days, season_months=months,
            train_seasons=sorted(train), clim=clim, bank=bank,
            bank_seasons=sorted({season_of(days[t], months) for t in picks}),
            anomalies=anomalies,
            currents=currents or schematic_currents(grid, land),
            winds=winds or schematic_winds(grid),
        )

    # ----------------------------------------------------------------- lookup
    @property
    def days(self) -> list[date]:
        return [np.datetime64(t, "D").astype(object) for t in self.ds["time"].values]

    def index_of(self, day: date) -> int:
        try:
            return self.days.index(day)
        except ValueError:
            raise ValueError(f"{day} is not in the dataset") from None

    def anomaly(self, season: int, day: date) -> np.ndarray:
        lib = self.anomalies[season]
        dos = day_of_season(day, self.season_months)
        key = min(lib, key=lambda k: abs(k - dos))  # nearest available day of that season
        return lib[key]

    # --------------------------------------------------------------- scenarios
    def scenarios(
        self, issue: date, n_days: int, n_members: int, rng: np.random.Generator, time_step_hours: float = 24.0
    ) -> ScenarioSet:
        t = self.index_of(issue)
        days = self.days
        lo = t - self.history_days + 1
        if lo < 0 or (days[t] - days[lo]).days != self.history_days - 1 or \
                season_of(days[lo], self.season_months) != season_of(days[t], self.season_months):
            raise ValueError(f"not enough contiguous in-season history before {issue}")
        for h in range(n_days):
            if season_of(issue + timedelta(days=h), self.season_months) is None:
                raise ValueError(f"scenario day {issue + timedelta(days=h)} is outside the season")

        seq = SequenceDataset(self.ds, [], self.history_days, self.lead_days, self.season_months)
        pred = np.asarray(self.predictor(seq.inputs(t)[None], [t]), float)[0]
        obs = self.ds["ice_concentration"].values
        land = self.ds["land_mask"].values.astype(bool)
        shape = land.shape
        K, H = n_members, self.lead_days
        conc = np.empty((K, n_days, *shape), np.float32)
        conc[:, 0] = obs[t]
        source = ["observed"]
        hh = min(H, n_days - 1)
        if hh:
            conc[:, 1: hh + 1] = residual_ensemble(pred, self.bank, K, rng)[:, :hh]
            source += ["forecast"] * hh
        clim_seasons: list[int] = []
        if n_days - 1 > H:
            clim_seasons = rng.choice(self.train_seasons, size=K).tolist()
            for h in range(H + 1, n_days):
                day = issue + timedelta(days=h)
                mu = self.clim.mean_for(day)
                for k, s in enumerate(clim_seasons):
                    conc[k, h] = np.clip(mu + self.anomaly(s, day), 0.0, 1.0)
            source += ["climatology"] * (n_days - 1 - H)
        conc[:, :, land] = np.nan

        cx, cy = self.currents
        wx, wy = self.winds
        mode = self.ds.attrs.get("execution_mode", "real")
        return ScenarioSet(
            grid=grid_of(self.ds), start=datetime(issue.year, issue.month, issue.day),
            time_step_hours=time_step_hours, land=land, conc=conc,
            current_x=np.broadcast_to(cx, (n_days, *shape)).copy(),
            current_y=np.broadcast_to(cy, (n_days, *shape)).copy(),
            execution_mode=mode,
            description=(f"Forecast issued {issue}: {hh} forecast day(s) from the model + residual bank, "
                         f"{max(0, n_days - 1 - H)} climatology-anomaly day(s); {K} joint members ({mode})."),
            wind_x=np.broadcast_to(wx, (n_days, *shape)).copy(),
            wind_y=np.broadcast_to(wy, (n_days, *shape)).copy(),
            layer_source=source,
            meta={"issue": issue.isoformat(), "forecast_days": hh, "climatology_seasons": clim_seasons},
        )
