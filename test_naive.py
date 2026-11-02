"""The D-7 rule, hand-checked on both 2025 clock changes and the weeks after them.

Each SP's demand is a code, 100 * (day of month) + SP, so a test can see which
row the naive copied: 3005 means "30th, SP 5".
"""
from datetime import date

import pandas as pd
import pytest

from gbtime import sp_grid
from naive import seasonal_naive


def coded_panel(start, end, drop=()):
    g = sp_grid(start, end)
    g["demand_mw"] = (100 * g["settlement_date"].dt.day + g["settlement_period"]).astype(float)
    g["publish_time"] = g["start_utc"] + pd.Timedelta(minutes=40)
    for day, sp in drop:
        g.loc[(g["settlement_date"] == pd.Timestamp(day)) & (g["settlement_period"] == sp), "demand_mw"] = float("nan")
    return g


@pytest.fixture(scope="module")
def spring():
    p = coded_panel(date(2025, 3, 9), date(2025, 4, 13))
    return pd.concat([p, seasonal_naive(p)], axis=1)


@pytest.fixture(scope="module")
def autumn():
    p = coded_panel(date(2025, 10, 5), date(2025, 11, 9))
    return pd.concat([p, seasonal_naive(p)], axis=1)


def naive_at(panel, day, sp):
    row = panel[(panel["settlement_date"] == pd.Timestamp(day)) & (panel["settlement_period"] == sp)]
    return row["naive"].item()


@pytest.mark.parametrize(
    "day, sp, expected",
    [
        pytest.param("2025-03-26", 20, 1920, id="ordinary: same SP last week"),
        pytest.param("2025-03-30", 2, 2302, id="46-SP day: 00:30 before the jump"),
        pytest.param("2025-03-30", 3, 2305, id="46-SP day: SP3 is 02:00, copies SP5"),
        pytest.param("2025-03-30", 46, 2348, id="46-SP day: SP46 is 23:30, copies SP48"),
        pytest.param("2025-04-06", 3, 2303, id="week after: 01:00 missing on D-7, so D-14"),
        pytest.param("2025-04-06", 4, 2304, id="week after: 01:30 missing on D-7, so D-14"),
        pytest.param("2025-04-06", 5, 3003, id="week after: 02:00 is SP3 on D-7"),
        pytest.param("2025-04-06", 48, 3046, id="week after: 23:30 is SP46 on D-7"),
    ],
)
def test_spring(spring, day, sp, expected):
    assert naive_at(spring, day, sp) == pytest.approx(expected)


@pytest.mark.parametrize(
    "day, sp, expected",
    [
        pytest.param("2025-10-26", 3, 1903, id="50-SP day: first 01:00 copies SP3"),
        pytest.param("2025-10-26", 5, 1903, id="50-SP day: second 01:00 also copies SP3"),
        pytest.param("2025-10-26", 7, 1905, id="50-SP day: SP7 is 02:00, copies SP5"),
        pytest.param("2025-10-26", 50, 1948, id="50-SP day: SP50 is 23:30, copies SP48"),
        pytest.param("2025-11-02", 3, (2603 + 2605) / 2, id="week after: 01:00 happened twice on D-7, average"),
        pytest.param("2025-11-02", 4, (2604 + 2606) / 2, id="week after: 01:30 happened twice on D-7, average"),
        pytest.param("2025-11-02", 5, 2607, id="week after: 02:00 is SP7 on D-7"),
        pytest.param("2025-11-02", 48, 2650, id="week after: 23:30 is SP50 on D-7"),
    ],
)
def test_autumn(autumn, day, sp, expected):
    assert naive_at(autumn, day, sp) == pytest.approx(expected)


def test_missing_data_gives_nan_not_a_fallback():
    """D-14 is only for clock times that never happened. Missing data must show as missing."""
    p = coded_panel(date(2025, 6, 1), date(2025, 6, 18), drop=[("2025-06-11", 20)])
    p = pd.concat([p, seasonal_naive(p)], axis=1)
    assert pd.isna(naive_at(p, "2025-06-18", 20))
    assert naive_at(p, "2025-06-18", 21) == pytest.approx(1121)


def test_a_missing_copy_of_the_repeated_hour_gives_nan():
    """On D-7 the clocks went back, so 01:00 happened twice. With one copy missing, the
    naive is missing too: it never averages around a gap using the copy that's left."""
    p = coded_panel(date(2025, 10, 19), date(2025, 11, 2), drop=[("2025-10-26", 5)])
    p = pd.concat([p, seasonal_naive(p)], axis=1)
    assert pd.isna(naive_at(p, "2025-11-02", 3))                              # 01:00: a source is gone
    assert naive_at(p, "2025-11-02", 4) == pytest.approx((2604 + 2606) / 2)   # 01:30: untouched


def test_naive_reports_when_its_source_was_published(spring):
    row = spring[(spring["settlement_date"] == pd.Timestamp("2025-03-26")) & (spring["settlement_period"] == 20)]
    source = spring[(spring["settlement_date"] == pd.Timestamp("2025-03-19")) & (spring["settlement_period"] == 20)]
    assert row["naive_info"].item() == source["publish_time"].item()
