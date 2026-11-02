"""The leak test: poison everything published after τ; the features at τ must not move.

If a feature at τ changes when post-τ data changes, that feature is reading the future.
The positive controls show the test has teeth.
"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

import backtest
import gbm
import gbtime
import synthetic

START, END = date(2024, 1, 1), date(2024, 3, 31)
DAY = pd.Timestamp("2024-03-15")
POISON = 1e9


@pytest.fixture(scope="module")
def data():
    return synthetic.make(START, END)


def features_for(outturn, wx_act, wx_fc):
    panel = backtest.build_panel(outturn, START, END)
    X_train, _, X_test, info_test = backtest.features(panel, wx_act, wx_fc)
    rows = (panel["settlement_date"] == DAY).to_numpy()
    return X_train[rows], X_test[rows], info_test[rows]


def poison_after(df, when, time_col, cols):
    df = df.copy()
    df.loc[df[time_col] > when, cols] = POISON
    return df


def test_test_features_ignore_everything_published_after_tau(data):
    tau = gbtime.issue_time(DAY)
    _, before, info = features_for(data.outturn, data.wx_act, data.wx_fc)
    _, after, _ = features_for(
        poison_after(data.outturn, tau, "publish_time", ["demand_mw"]),
        poison_after(data.wx_act, tau, "info_time", gbm.WEATHER_VARS),
        poison_after(data.wx_fc, tau, "info_time", gbm.WEATHER_VARS),
    )
    pd.testing.assert_frame_equal(before, after)
    for version, feats in gbm.FEATURE_SETS.items():
        assert (gbm.row_info(info, feats) <= tau).all(), version
    assert (gbm.row_info(info, gbm.FEATURE_SETS["v2"]) >= info["morning"]).all()  # v2 is charged for its morning


def test_positive_control_last_week_moves_the_naive(data):
    """Poison D-7 (published long before τ): the naive must move, so the test above has teeth."""
    out = data.outturn.copy()
    out.loc[out["settlement_date"] == DAY - pd.Timedelta(days=7), "demand_mw"] = POISON
    _, X, _ = features_for(out, data.wx_act, data.wx_fc)
    assert (X["naive"] == POISON).all()


def test_positive_control_reanalysis_at_d_is_the_future(data):
    """Training rows use the real weather on D. Fine for training; at τ it would be a leak."""
    tau = gbtime.issue_time(DAY)
    train_before, _, _ = features_for(data.outturn, data.wx_act, data.wx_fc)
    train_after, _, _ = features_for(data.outturn, poison_after(data.wx_act, tau, "info_time", gbm.WEATHER_VARS),
                                     data.wx_fc)
    assert (train_after["temp"] == POISON).all() and not (train_before["temp"] == POISON).any()


def test_positive_control_this_morning_moves_v2s_feature(data):
    """Poison D-1 at 07:30, published long before τ: the morning number must move by 1/16 of it."""
    out = data.outturn.copy()
    hit = (out["settlement_date"] == DAY - pd.Timedelta(days=1)) & (out["settlement_period"] == 16)
    out.loc[hit, "demand_mw"] += 1600.0
    _, before, _ = features_for(data.outturn, data.wx_act, data.wx_fc)
    _, after, _ = features_for(out, data.wx_act, data.wx_fc)
    assert (after["morning_anom"] - before["morning_anom"]).to_numpy() == pytest.approx(np.full(len(after), 100.0))


# --- the morning number, hand-checked. The naive misses by +50 MW every morning half-hour
#     and by +999 MW later in the day, so only the 00:00-08:00 window may show. ---

def morning_panel(start, end):
    g = gbtime.sp_grid(start, end)
    g["naive"] = 1000.0
    g["demand_mw"] = np.where(g["tod"] < gbm.MORNING_END_TOD, 1050.0, 1999.0)
    g["publish_time"] = g["start_utc"] + pd.Timedelta(minutes=45)
    g["naive_info"] = g["start_utc"] - pd.Timedelta(days=7)
    return g


def bump(panel, day, sp, mw):
    panel.loc[(panel["settlement_date"] == pd.Timestamp(day)) & (panel["settlement_period"] == sp), "demand_mw"] += mw


@pytest.mark.parametrize(
    "day, sp, n",
    [
        pytest.param("2024-01-15", 9, 16, id="ordinary morning: 16 half-hours"),
        pytest.param("2024-03-31", 3, 14, id="clocks forward: 14 half-hours (no 01:00-01:59)"),
        pytest.param("2024-10-27", 5, 18, id="clocks back: 18 half-hours (01:00-01:59 twice)"),
    ],
)
def test_morning_anomaly_averages_00_to_08_only(day, sp, n):
    p = morning_panel(pd.Timestamp(day) - pd.Timedelta(days=1), pd.Timestamp(day) + pd.Timedelta(days=1))
    bump(p, day, sp, 10.0 * n)  # one half-hour 10*n MW higher moves the average by exactly 10
    value, info = gbm.morning_anomaly(p)
    assert value[pd.Timestamp(day)] == pytest.approx(60.0)
    assert info[pd.Timestamp(day)] <= gbtime.issue_time(pd.Timestamp(day) + pd.Timedelta(days=1))


@pytest.mark.parametrize(
    "late_by, expect_nan",
    [
        pytest.param(pd.Timedelta(0), False, id="published exactly at τ: counts"),
        pytest.param(pd.Timedelta(minutes=1), True, id="published a minute after τ: missing"),
    ],
)
def test_morning_half_hour_not_out_by_tau_makes_the_morning_missing(late_by, expect_nan):
    p = morning_panel("2024-01-14", "2024-01-16")
    tau = gbtime.issue_time(pd.Timestamp("2024-01-16"))  # 15 January, 09:30
    hit = (p["settlement_date"] == pd.Timestamp("2024-01-15")) & (p["settlement_period"] == 16)
    p.loc[hit, "publish_time"] = tau + late_by
    value, _ = gbm.morning_anomaly(p)
    assert pd.isna(value[pd.Timestamp("2024-01-15")]) == expect_nan


def test_a_missing_morning_half_hour_makes_the_morning_missing():
    p = morning_panel("2024-01-14", "2024-01-16")
    p.loc[(p["settlement_date"] == pd.Timestamp("2024-01-15")) & (p["settlement_period"] == 4), "demand_mw"] = np.nan
    value, _ = gbm.morning_anomaly(p)
    assert pd.isna(value[pd.Timestamp("2024-01-15")]) and value[pd.Timestamp("2024-01-14")] == pytest.approx(50.0)
