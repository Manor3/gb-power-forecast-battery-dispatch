"""Rolling-origin backtest: issue every day at τ, refit the GBM once a month.

Pure: no network and no files, so the tests can run it on synthetic data.
The run script does the fetching, caching and storing.
"""
import numpy as np
import pandas as pd

import gbm
import gbtime
import pit
from naive import seasonal_naive

FORECAST_COLUMNS = ["model", "issue_time", "settlement_date", "settlement_period", "forecast_mw", "source_time"]
MIN_TRAIN_ROWS = 1_000

# An outturn with no publish time is assumed published 15 minutes after its SP ends,
# Elexon's stated target for INDO.
INDO_ASSUMED_LAG = pd.Timedelta(minutes=45)


def build_panel(outturn, start, end):
    """Every SP from start to end: demand where we have it (NaN where not), plus the naive."""
    grid = gbtime.sp_grid(start, end)
    panel = grid.merge(outturn[["settlement_date", "settlement_period", "demand_mw", "publish_time"]],
                       on=["settlement_date", "settlement_period"], how="left")
    panel["demand_mw"] = panel["demand_mw"].astype(float)
    no_time = panel["demand_mw"].notna() & panel["publish_time"].isna()
    panel.loc[no_time, "publish_time"] = panel.loc[no_time, "start_utc"] + INDO_ASSUMED_LAG
    return pd.concat([panel, seasonal_naive(panel)], axis=1)


def features(panel, wx_act, wx_fc, issue_at=gbtime.ISSUE_TIME):
    """Training rows see reanalysis; test rows see forecasts. Last week is reanalysis in both."""
    X_train, info_train = gbm.build_features(panel, wx_target=wx_act, wx_lastweek=wx_act, issue_at=issue_at)
    X_test, info_test = gbm.build_features(panel, wx_target=wx_fc, wx_lastweek=wx_act, issue_at=issue_at)
    return X_train, info_train, X_test, info_test


def _rows(model, tau, day, sps, values, source_time=pd.NaT):
    return pd.DataFrame({
        "model": model,
        "issue_time": tau,
        "settlement_date": day,
        "settlement_period": np.asarray(sps),
        "forecast_mw": np.asarray(values, dtype=float),
        "source_time": source_time,
    })


def run_backtest(outturn, wx_act, wx_fc, ndf, test_days, train_start, version,
                 params=gbm.PARAMS, issue_at=gbtime.ISSUE_TIME, diagnostic=True):
    """Forecasts for every SP of test_days from the naive, NESO and the GBM.

    outturn: INDO history from at least 14 days before train_start (for the naive and lags)
    wx_act:  half-hourly reanalysis covering the whole history; wx_fc: forecasts for test_days
    ndf:     every NDF vintage for test_days
    Returns (forecasts, panel, log). Forecasts are long: one row per model and SP; NaN
    forecasts are dropped and logged instead.
    """
    log = []
    test_days = pd.DatetimeIndex(sorted({pd.Timestamp(d) for d in test_days})).as_unit("ns")
    panel = build_panel(outturn, outturn["settlement_date"].min(), test_days.max())
    X_train, info_train, X_test, info_test = features(panel, wx_act, wx_fc, issue_at)

    feats = gbm.FEATURE_SETS[version]
    wx_feats = [f for f in feats if f in gbm.WEATHER]
    info_fit, info_fc = gbm.row_info(info_train, feats), gbm.row_info(info_test, feats)
    target = panel["demand_mw"] - panel["naive"]
    eligible = (panel["settlement_date"] >= pd.Timestamp(train_start)) & target.notna()
    taus = gbtime.issue_times(test_days, issue_at)
    rows_of = {pd.Timestamp(k): v for k, v in panel.groupby("settlement_date").indices.items()}  # day -> rows
    name = f"gbm_{version}"
    out = []

    for _, month in test_days.to_series().groupby(test_days.to_period("M")):
        refit_tau = taus[month.iloc[0]]
        fit_rows = eligible & pit.trainable(panel["publish_time"], info_fit, refit_tau)
        if fit_rows.sum() < MIN_TRAIN_ROWS:
            raise ValueError(f"only {int(fit_rows.sum())} trainable rows at {refit_tau}: is the history long enough?")
        model = gbm.fit(X_train.loc[fit_rows, feats], target[fit_rows], params)
        newest = panel.loc[fit_rows, "start_utc"].max()
        log.append(dict(date=month.iloc[0].date().isoformat(), stage="refit",
                        message=f"{name}: fitted at {refit_tau:%Y-%m-%d %H:%M}Z on {int(fit_rows.sum())} rows, newest {newest:%Y-%m-%d %H:%M}Z"))

        for day in month:
            tau, pos, d = taus[day], rows_of[day], day.date().isoformat()
            sps = panel["settlement_period"].to_numpy()[pos]
            naive_today = panel["naive"].to_numpy()[pos]
            pit.assert_no_leak(panel["naive_info"].iloc[pos], tau, f"naive {d}")
            pit.assert_no_leak(info_fc.iloc[pos], tau, f"{name} {d}")

            missing = np.isnan(naive_today)
            if missing.any():
                log.append(dict(date=d, stage="naive", message=f"no D-7 outturn for SPs {sps[missing].tolist()}"))
            out.append(_rows("naive", tau, day, sps[~missing], naive_today[~missing]))

            pred = naive_today + model.predict(X_test.iloc[pos][feats])
            if wx_feats:
                no_wx = X_test.iloc[pos][wx_feats].isna().any(axis=1).to_numpy()
                if no_wx.any():
                    log.append(dict(date=d, stage=name, message=f"no point-in-time weather for SPs {sps[no_wx].tolist()}"))
                pred[no_wx] = np.nan
            ok = ~np.isnan(pred)
            out.append(_rows(name, tau, day, sps[ok], pred[ok]))

            if diagnostic and wx_feats:  # deliberately leaks the real weather: a diagnostic, never a contestant
                actual = naive_today + model.predict(X_train.iloc[pos][feats])
                ok = ~np.isnan(actual)
                out.append(_rows(f"{name}_actualwx", tau, day, sps[ok], actual[ok]))

    neso = pit.pick_vintage(ndf, taus)
    for day, g in neso.groupby("settlement_date"):
        out.append(_rows("neso", taus[day], day, g["settlement_period"], g["demand_mw"], g["publish_time"].to_numpy()))
    no_neso = taus.index.difference(pd.DatetimeIndex(neso["settlement_date"].unique()))
    for day in no_neso:
        log.append(dict(date=day.date().isoformat(), stage="neso", message="no NDF vintage published by τ"))

    forecasts = pd.concat([f for f in out if len(f)], ignore_index=True)[FORECAST_COLUMNS]
    forecasts["source_time"] = pd.to_datetime(forecasts["source_time"], utc=True)
    return forecasts, panel, log
