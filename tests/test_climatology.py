from datetime import date, timedelta

import numpy as np
import pytest

from antarctic_routing.preprocessing.climatology import (
    Climatology,
    chronological_split,
    day_of_season,
    season_of,
)

SEASON = [11, 12, 1, 2]


@pytest.mark.parametrize(
    "d,expected",
    [(date(2026, 11, 1), 2026), (date(2026, 12, 31), 2026), (date(2027, 1, 15), 2026),
     (date(2027, 2, 28), 2026), (date(2027, 6, 1), None)],
)
def test_season_spans_new_year(d, expected):
    assert season_of(d, SEASON) == expected


def test_non_wrapping_season():
    assert season_of(date(2026, 2, 1), [1, 2, 3]) == 2026


def test_day_of_season_is_continuous_across_new_year():
    assert day_of_season(date(2026, 11, 1), SEASON) == 0
    assert day_of_season(date(2026, 12, 31), SEASON) == 60
    assert day_of_season(date(2027, 1, 1), SEASON) == 61


def test_chronological_split_is_ordered_and_disjoint():
    split = chronological_split([2015, 2012, 2013, 2014, 2016, 2017, 2018], n_val=2, n_test=2)
    assert split == {"train": [2012, 2013, 2014], "val": [2015, 2016], "test": [2017, 2018]}


def test_chronological_split_needs_training_data():
    with pytest.raises(ValueError):
        chronological_split([2020, 2021], n_val=1, n_test=1)


def _synthetic_series(seasons, offset_by_season=None):
    dates, fields = [], []
    for s in seasons:
        start = date(s, 11, 1)
        for k in range(120):
            d = start + timedelta(days=k)
            base = 0.8 - 0.005 * k  # seasonal decline
            off = (offset_by_season or {}).get(s, 0.0)
            dates.append(d)
            fields.append(np.full((2, 2), base + off))
    return dates, np.stack(fields)


def test_climatology_mean_and_anomaly_hand_values():
    dates, conc = _synthetic_series([2010, 2011], offset_by_season={2010: -0.1, 2011: 0.1})
    clim = Climatology.fit(conc, dates, train_seasons=[2010, 2011], season_months=SEASON, window_days=0)
    # mean of (base-0.1) and (base+0.1) == base
    assert clim.mean_for(date(2026, 11, 1))[0, 0] == pytest.approx(0.8)
    assert clim.std_for(date(2026, 11, 1))[0, 0] == pytest.approx(0.1)
    anomaly = clim.anomaly(np.full((2, 2), 0.95), date(2026, 11, 1))
    assert anomaly[0, 0] == pytest.approx(0.15)
    z = clim.standardized(np.full((2, 2), 0.95), date(2026, 11, 1))
    assert z[0, 0] == pytest.approx(0.15 / (0.1 + clim.eps))


def test_climatology_ignores_non_training_seasons_no_leakage():
    dates, conc = _synthetic_series([2010, 2011, 2012])
    clim_a = Climatology.fit(conc, dates, train_seasons=[2010, 2011], season_months=SEASON)
    conc_b = conc.copy()
    conc_b[[i for i, d in enumerate(dates) if season_of(d, SEASON) == 2012]] = 0.0
    clim_b = Climatology.fit(conc_b, dates, train_seasons=[2010, 2011], season_months=SEASON)
    assert np.array_equal(clim_a.mean, clim_b.mean, equal_nan=True)


def test_window_smoothing_fills_sparse_days():
    dates, conc = _synthetic_series([2010])
    keep = [i for i, d in enumerate(dates) if d != date(2010, 11, 20)]
    clim = Climatology.fit(conc[keep], [dates[i] for i in keep], [2010], SEASON, window_days=3)
    assert np.isfinite(clim.mean_for(date(2030, 11, 20))).all()
