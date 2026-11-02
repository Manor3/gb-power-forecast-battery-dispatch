"""Weather interpolation, site weighting and legality at τ."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

import weather
from gbtime import issue_time, sp_grid


def hours(*stamps):
    return pd.DatetimeIndex(pd.to_datetime(list(stamps), utc=True)).as_unit("ns")


def test_instantaneous_values_interpolate_to_the_sp_midpoint():
    hourly = pd.DataFrame({"temp": [0.0, 10.0]}, index=hours("2025-01-15 00:00", "2025-01-15 01:00"))
    hh = weather.to_half_hourly(hourly, hours("2025-01-15 00:00", "2025-01-15 00:30"))
    assert hh["temp"].tolist() == pytest.approx([2.5, 7.5])  # midpoints 00:15 and 00:45


def test_hour_means_are_centred_on_their_hour_first():
    """Irradiance stamped 01:00 is the mean over 00:00-01:00, so it belongs to 00:30."""
    hourly = pd.DataFrame({"ghi": [100.0, 200.0]}, index=hours("2025-06-18 01:00", "2025-06-18 02:00"))
    hh = weather.to_half_hourly(hourly, hours("2025-06-18 00:30", "2025-06-18 01:00"))
    assert hh["ghi"].tolist() == pytest.approx([125.0, 175.0])  # midpoints 00:45 and 01:15


def test_a_missing_hour_is_not_bridged():
    hourly = pd.DataFrame({"temp": [0.0, np.nan, 20.0]},
                          index=hours("2025-01-15 00:00", "2025-01-15 01:00", "2025-01-15 02:00"))
    hh = weather.to_half_hourly(hourly, hours("2025-01-15 00:30", "2025-01-15 01:00"))
    assert hh["temp"].isna().all()


def test_national_average_is_weighted_and_strict():
    t = hours("2025-01-15 00:00", "2025-01-15 01:00")
    rows = []
    for site, (_, _, people, solar) in weather.SITES.items():
        for var in weather.WEIGHTING:
            value = 10.0 if site == "london" else 0.0
            rows += [dict(time_utc=t[0], site=site, variable=var, value=value),
                     dict(time_utc=t[1], site=site, variable=var, value=np.nan if site == "glasgow" else value)]
    nat = weather.national(pd.DataFrame(rows))
    people = sum(s[2] for s in weather.SITES.values())
    solar = sum(s[3] for s in weather.SITES.values())
    assert nat.loc[t[0], "temp"] == pytest.approx(10.0 * 9.5 / people)
    assert nat.loc[t[0], "ghi"] == pytest.approx(10.0 * 1.0 / solar)
    assert nat.loc[t[1]].isna().all()  # one site missing: NaN, never a silent reweighting


def test_legality_at_tau():
    """Every hour of D forecast 48 h ahead is legal at τ; reanalysis is legal for D-7 only."""
    day = date(2025, 6, 18)
    tau = issue_time(day)
    sps_d = pd.DatetimeIndex(sp_grid(day, day)["start_utc"])
    sps_lw = pd.DatetimeIndex(sp_grid(date(2025, 6, 11), date(2025, 6, 11))["start_utc"])
    assert (weather.info_time("forecast", sps_d) <= tau).all()
    assert (weather.info_time("reanalysis", sps_lw) <= tau).all()
    assert (weather.info_time("reanalysis", sps_d) > tau).all()
