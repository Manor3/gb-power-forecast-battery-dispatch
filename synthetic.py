"""Synthetic GB-like data for the tests. No network.

Built so a wiring mistake shows up in the results:
- weather drives demand, so a GBM whose weather features are misaligned can't beat the naive;
- NESO publishes twice: 09:05 (noisy) and 12:05 (nearly perfect). A vintage rule that lets
  the 12:05 one through would make NESO look far better than it should.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

import gbtime
import weather


def _ar1(rng, n, phi, sigma):
    x = np.zeros(n)
    shocks = rng.normal(0, sigma, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + shocks[i]
    return x


def make(start, end, seed=0):
    rng = np.random.default_rng(seed)

    # hourly "real" weather, with multi-day regimes so last week's weather is a poor guide
    hours = pd.date_range(pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=1),
                          pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=2), freq="h").as_unit("ns")
    n = len(hours)
    doy, hod = hours.dayofyear.to_numpy(), hours.hour.to_numpy()
    season = -np.cos(2 * np.pi * (doy - 20) / 365)  # -1 mid-January, +1 mid-July
    temp = 11 + 6 * season + 3 * np.sin(2 * np.pi * (hod - 9) / 24) + _ar1(rng, n, 0.995, 0.3)
    cloud = np.clip(55 + _ar1(rng, n, 0.98, 6), 0, 100)
    sun = np.clip(np.sin(np.pi * (hod - 6) / 12), 0, None) * (550 + 250 * season)
    ghi = sun * (1 - 0.75 * cloud / 100)
    wind = np.abs(6 + _ar1(rng, n, 0.99, 0.5))
    real = pd.DataFrame({"temp": temp, "ghi": ghi, "wind": wind, "cloud": cloud}, index=hours)

    # the 48-hour-ahead forecast: the real weather plus a forecast error
    fc = real.copy()
    fc["temp"] += _ar1(rng, n, 0.9, 0.4)
    fc["ghi"] = np.clip(fc["ghi"] * (1 + _ar1(rng, n, 0.9, 0.05)), 0, None)
    fc["wind"] = np.abs(fc["wind"] + _ar1(rng, n, 0.9, 0.3))
    fc["cloud"] = np.clip(fc["cloud"] + _ar1(rng, n, 0.9, 3), 0, 100)

    grid = gbtime.sp_grid(start, end)
    wx_act = weather.to_half_hourly(real, grid["start_utc"])
    wx_act["info_time"] = weather.info_time("reanalysis", wx_act.index)
    wx_fc = weather.to_half_hourly(fc, grid["start_utc"])
    wx_fc["info_time"] = weather.info_time("forecast", wx_fc.index)

    # demand: a daily shape, the calendar, heating, embedded solar and wind, and noise
    w = wx_act.reindex(grid["start_utc"])
    t, g, wi = (w[c].to_numpy(dtype=float) for c in ("temp", "ghi", "wind"))
    kind = gbtime.day_type(grid["settlement_date"])
    shape = -np.cos(2 * np.pi * (grid["tod"].to_numpy() - 9) / 48) + 0.4 * np.exp(-((grid["tod"].to_numpy() - 35) ** 2) / 8)
    demand = (27_000 + 5_000 * shape
              - 2_500 * (kind == "weekend") - 4_000 * (kind == "bank_holiday") - 5_000 * (kind == "christmas")
              + 450 * np.clip(15 - t, 0, None) - 9 * g - 120 * wi
              + rng.normal(0, 250, len(grid)))
    outturn = pd.DataFrame({
        "settlement_date": grid["settlement_date"],
        "settlement_period": grid["settlement_period"],
        "demand_mw": demand,
        "publish_time": grid["start_utc"] + pd.Timedelta(minutes=40),
    })

    # NESO: two vintages per target day, published on D-1
    rows = []
    for day, g_day in outturn.groupby("settlement_date"):
        before = pd.Timestamp(day - pd.Timedelta(days=1)).tz_localize(gbtime.TZ)
        for hhmm, noise in (("09:05", 600), ("12:05", 50)):
            published = (before + pd.Timedelta(hhmm + ":00")).tz_convert("UTC")
            rows.append(pd.DataFrame({
                "settlement_date": day,
                "settlement_period": g_day["settlement_period"].to_numpy(),
                "publish_time": published,
                "demand_mw": g_day["demand_mw"].to_numpy() + rng.normal(0, noise, len(g_day)),
            }))
    ndf = pd.concat(rows, ignore_index=True)
    ndf["publish_time"] = ndf["publish_time"].astype("datetime64[ns, UTC]")

    return SimpleNamespace(outturn=outturn, wx_act=wx_act, wx_fc=wx_fc, ndf=ndf, real_hourly=real, forecast_hourly=fc)
