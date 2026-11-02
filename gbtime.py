"""GB clock and calendar: the settlement-period grid, issue times and day types.

Everything that depends on the GB clock lives here, so clock changes are handled once.
Settlement periods count half-hours from local midnight. On the 46- and 50-period days
SP numbers stop matching clock time, so anything about behaviour uses `tod`, the local
half-hour of the day (0 = 00:00, 47 = 23:30).
"""
from datetime import datetime, time, timedelta

import holidays
import numpy as np
import pandas as pd

from elexon import expected_periods

TZ = "Europe/London"

# Issue time τ: 09:30 on D-1. NESO publishes its day-ahead forecast between 09:00 and
# 09:15, so at 09:00 sharp there would be no NESO forecast to grade. Nothing in the GBM
# changes between 09:00 and 09:30: its newest input is demand on D-2.
ISSUE_TIME = time(9, 30)


def issue_time(day, at=ISSUE_TIME):
    """τ for settlement date `day`: clock time `at` on the day before, as a UTC Timestamp."""
    day = pd.Timestamp(day).date()
    local = pd.Timestamp(datetime.combine(day - timedelta(days=1), at)).tz_localize(TZ)
    return local.tz_convert("UTC").as_unit("ns")


def issue_times(days, at=ISSUE_TIME):
    """Series: settlement date (midnight Timestamp) -> τ."""
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in days]).as_unit("ns")
    return pd.Series([issue_time(d, at) for d in index], index=index, name="tau")


def sp_grid(start, end):
    """Every settlement period of settlement dates start..end inclusive.

    Columns: settlement_date (midnight, no time zone), settlement_period, start_utc, tod.
    """
    days = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D").as_unit("ns")
    n = np.array([expected_periods(d.date()) for d in days])
    midnight_utc = days.tz_localize(TZ).tz_convert("UTC")  # UK midnight is never ambiguous
    grid = pd.DataFrame({"settlement_date": days.repeat(n), "midnight_utc": midnight_utc.repeat(n)})
    grid["settlement_period"] = grid.groupby("settlement_date").cumcount() + 1
    grid["start_utc"] = grid["midnight_utc"] + pd.to_timedelta(30 * (grid["settlement_period"] - 1), unit="min")
    local = grid["start_utc"].dt.tz_convert(TZ)
    grid["tod"] = local.dt.hour * 2 + local.dt.minute // 30
    return grid.drop(columns="midnight_utc")


def bank_holidays(years):
    """England and Wales bank holidays, including one-offs such as the 2023 coronation."""
    return pd.DatetimeIndex(sorted(holidays.country_holidays("GB", subdiv="ENG", years=list(years))))


def day_type(dates):
    """One of christmas (24 Dec to 1 Jan), bank_holiday, weekend, weekday, in that order of precedence."""
    d = pd.DatetimeIndex(dates)
    hols = bank_holidays(range(d.year.min(), d.year.max() + 1))
    xmas = ((d.month == 12) & (d.day >= 24)) | ((d.month == 1) & (d.day == 1))
    return np.select([xmas, d.isin(hols), d.dayofweek >= 5], ["christmas", "bank_holiday", "weekend"], "weekday")


def utc_offset_hours(dates):
    """Local clock minus UTC, in hours, at noon on each date: 0 in winter, 1 in summer."""
    d = pd.DatetimeIndex(dates)
    uniq = d.unique()
    noon = (uniq + pd.Timedelta(hours=12)).tz_localize(TZ)
    offsets = pd.Series([t.utcoffset().total_seconds() / 3600 for t in noon], index=uniq)
    return offsets.reindex(d).to_numpy()
