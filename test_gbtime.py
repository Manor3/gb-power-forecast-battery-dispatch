"""Tests for gbtime.py: the grid, issue times and day types."""
from datetime import date

import pandas as pd
import pytest

from elexon import sp_start_utc
from gbtime import day_type, issue_time, sp_grid, utc_offset_hours


@pytest.mark.parametrize(
    "day, utc",
    [
        pytest.param(date(2025, 1, 15), "2025-01-14 09:30", id="winter: 09:30 GMT"),
        pytest.param(date(2025, 6, 18), "2025-06-17 08:30", id="summer: 09:30 BST is 08:30 UTC"),
        pytest.param(date(2025, 3, 30), "2025-03-29 09:30", id="target is the 46-SP day; D-1 still GMT"),
        pytest.param(date(2025, 3, 31), "2025-03-30 08:30", id="D-1 is the clocks-forward day, BST by 09:30"),
    ],
)
def test_issue_time(day, utc):
    assert issue_time(day) == pd.Timestamp(utc, tz="UTC")


def test_grid_tod_follows_the_clock_on_clock_change_days():
    g = sp_grid(date(2025, 3, 30), date(2025, 3, 30))
    assert g["tod"].tolist() == [0, 1] + list(range(4, 48))  # 01:00-01:59 never happened
    g = sp_grid(date(2025, 10, 26), date(2025, 10, 26))
    assert g["tod"].tolist() == [0, 1, 2, 3, 2, 3] + list(range(4, 48))  # 01:00-01:59 happened twice


def test_grid_matches_sp_start_utc_everywhere():
    g = sp_grid(date(2025, 3, 28), date(2025, 4, 2))
    expected = [pd.Timestamp(sp_start_utc(d.date(), sp)) for d, sp in zip(g["settlement_date"], g["settlement_period"])]
    assert g["start_utc"].tolist() == expected


@pytest.mark.parametrize(
    "day, kind",
    [
        pytest.param("2025-06-18", "weekday", id="wednesday"),
        pytest.param("2025-06-21", "weekend", id="saturday"),
        pytest.param("2025-08-25", "bank_holiday", id="late-summer-bank-holiday"),
        pytest.param("2023-05-08", "bank_holiday", id="coronation-one-off"),
        pytest.param("2025-12-27", "christmas", id="christmas-beats-weekend"),
        pytest.param("2026-01-01", "christmas", id="new-years-day-is-christmas-period"),
    ],
)
def test_day_type(day, kind):
    assert day_type([pd.Timestamp(day)])[0] == kind


def test_utc_offset():
    assert list(utc_offset_hours(pd.to_datetime(["2025-01-15", "2025-07-15"]))) == [0.0, 1.0]
