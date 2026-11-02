"""Gradient-boosting demand model: features and fitting.

Target: y - naive, the change since the same local half-hour last week. Trees can't
extrapolate, and INDO drifts down as embedded solar grows. The naive carries the level;
the features explain what changed since D-7.

Every feature row comes with its info time: the latest publish time among the numbers
it used. The backtest checks info time <= τ for every forecast it issues.

Frozen on the 2024 validation year: change anything here only while YEAR = 2024,
and give each attempt a new MODEL_VERSION.
"""
import lightgbm as lgb
import pandas as pd

import gbtime
from naive import shift_local

CALENDAR = ["tod", "dow", "doy", "is_hol", "is_hol_lw", "is_xmas", "is_xmas_lw", "dst_shift"]
DEMAND = ["naive", "chg_d2"]
WEATHER = ["temp", "d_temp", "ghi", "d_ghi", "wind", "d_wind", "cloud", "d_cloud"]
WEATHER_VARS = ["temp", "ghi", "wind", "cloud"]
MORNING = ["morning_anom"]

# "This morning": D-1 from 00:00 to 08:00 local. It ends 90 minutes before τ, so it's
# published in time even when Elexon runs late; anything not out by τ counts as missing.
MORNING_END_TOD = 16  # tod 16 = 08:00

FEATURE_SETS = {
    "v0": CALENDAR + DEMAND,                      # calendar and lags: must beat the naive
    "v1": CALENDAR + DEMAND + WEATHER,            # adds weather: must beat v0
    "v2": CALENDAR + DEMAND + WEATHER + MORNING,  # adds how far off the naive was this morning: must beat v1
}

PARAMS = dict(
    objective="regression",  # squared error: big misses cost more than proportionally
    learning_rate=0.05,
    n_estimators=600,
    num_leaves=31,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=0,
    deterministic=True,
    force_col_wise=True,
    verbose=-1,
)


def calendar_features(panel):
    """Calendar features for target day D and for D-7, the day the naive copies."""
    d = pd.DatetimeIndex(panel["settlement_date"])
    last_week = d - pd.Timedelta(days=7)
    kind, kind_lw = gbtime.day_type(d), gbtime.day_type(last_week)
    return pd.DataFrame({
        "tod": panel["tod"].to_numpy(),
        "dow": d.dayofweek,
        "doy": d.dayofyear,
        "is_hol": (kind == "bank_holiday").astype(int),
        "is_hol_lw": (kind_lw == "bank_holiday").astype(int),  # a holiday last week breaks the naive
        "is_xmas": (kind == "christmas").astype(int),
        "is_xmas_lw": (kind_lw == "christmas").astype(int),
        # +1 or -1 in the week after a clock change: sunset moves an hour against the naive
        "dst_shift": gbtime.utc_offset_hours(d) - gbtime.utc_offset_hours(last_week),
    }, index=panel.index)


def morning_anomaly(panel, issue_at=gbtime.ISSUE_TIME):
    """How far off the naive was each morning (00:00-08:00 local), in MW, as known at 09:30 that day.

    Keyed by the morning's own date d. Target day D = d + 1 uses it: d's 09:30 is D's τ.
    A half-hour not published by then counts as missing, and one missing half-hour makes
    the whole morning missing: an average around a gap would change what the number means.
    Returns (value, latest publish time used) per date.
    """
    m = panel[panel["tod"] < MORNING_END_TOD]
    tau = gbtime.issue_times(pd.DatetimeIndex(m["settlement_date"].unique()) + pd.Timedelta(days=1), issue_at)
    tau.index = tau.index - pd.Timedelta(days=1)  # key by the morning's own date
    published = m["publish_time"] <= m["settlement_date"].map(tau)
    miss = (m["demand_mw"] - m["naive"]).where(published)
    g = miss.groupby(m["settlement_date"])
    value = g.mean().where(g.count() == g.size())
    used = pd.concat([m["publish_time"].where(published), m["naive_info"]], axis=1).max(axis=1)
    return value, used.groupby(m["settlement_date"]).max()


def _weather_on_grid(panel, wx):
    """wx (indexed by SP start, UTC) lined up with panel's rows."""
    cols = WEATHER_VARS + ["info_time"]
    if wx is None:
        return pd.DataFrame(index=panel.index, columns=cols, dtype=float)
    return wx.reindex(pd.DatetimeIndex(panel["start_utc"]).as_unit("ns"))[cols].set_axis(panel.index)


def build_features(panel, wx_target, wx_lastweek, issue_at=gbtime.ISSUE_TIME):
    """Features for every row of panel, and each row's info_time.

    panel:       sp_grid rows with demand_mw, publish_time, naive, naive_info
    wx_target:   weather for the target half-hour (forecasts when testing, reanalysis when training)
    wx_lastweek: weather for D-7, always reanalysis (it's in the past at τ)
    Both weather frames: indexed by SP start (UTC), columns WEATHER_VARS + info_time.
    Returns X and info: per row, the latest publish time among its demand, weather and morning inputs.
    """
    X = calendar_features(panel)
    X["naive"] = panel["naive"]

    two = shift_local(panel, 2, ["demand_mw"], ["publish_time"])
    nine = shift_local(panel, 9, ["demand_mw"], ["publish_time"])
    X["chg_d2"] = two["demand_mw"] - nine["demand_mw"]  # how far off the naive was two days ago

    target = _weather_on_grid(panel, wx_target)
    lw_grid = pd.concat([panel[["settlement_date", "tod"]], _weather_on_grid(panel, wx_lastweek)], axis=1)
    lw_grid["info_time"] = pd.to_datetime(lw_grid["info_time"], utc=True)
    last_week = shift_local(lw_grid, 7, WEATHER_VARS, ["info_time"], fallback_days=14)
    for v in WEATHER_VARS:
        X[v] = target[v].astype(float)
        X[f"d_{v}"] = target[v].astype(float) - last_week[v]

    value, morning_info = morning_anomaly(panel, issue_at)
    yesterday = panel["settlement_date"] - pd.Timedelta(days=1)
    X["morning_anom"] = yesterday.map(value).astype(float)

    info = pd.DataFrame({
        "demand": pd.concat([panel["naive_info"], two["publish_time"], nine["publish_time"]], axis=1).max(axis=1),
        "weather": pd.concat([pd.to_datetime(target["info_time"], utc=True), last_week["info_time"]], axis=1).max(axis=1),
        "morning": pd.to_datetime(yesterday.map(morning_info), utc=True),
    }, index=panel.index)
    return X, info


def row_info(info, features):
    """Latest publish time among the inputs a feature set actually uses."""
    groups = ["demand"]
    groups += ["weather"] if any(f in WEATHER for f in features) else []
    groups += ["morning"] if any(f in MORNING for f in features) else []
    return info[groups].max(axis=1)


def fit(X, y, params=PARAMS):
    """Fit with LightGBM's native API (the LGBMRegressor wrapper also needs scikit-learn)."""
    p = dict(params)
    rounds = p.pop("n_estimators")
    return lgb.train(p, lgb.Dataset(X, label=y), num_boost_round=rounds)
