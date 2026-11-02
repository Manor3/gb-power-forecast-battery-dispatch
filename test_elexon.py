"""Tests for elexon.py that need no network."""
from datetime import date
import math

import pandas as pd
import pytest

import elexon
from elexon import expected_periods, missing_periods, parse_indo, parse_ndf, sp_start_utc


@pytest.mark.parametrize(
    "day, n",
    [
        pytest.param(date(2025, 6, 18), 48, id="ordinary-day"),
        pytest.param(date(2025, 3, 23), 48, id="sunday-before-clocks-forward"),
        pytest.param(date(2025, 3, 30), 46, id="clocks-forward-2025"),
        pytest.param(date(2025, 10, 26), 50, id="clocks-back-2025"),
        pytest.param(date(2026, 3, 29), 46, id="clocks-forward-2026"),
        pytest.param(date(2026, 10, 25), 50, id="clocks-back-2026"),
    ],
)
def test_expected_periods(day, n):
    assert expected_periods(day) == n


@pytest.mark.parametrize(
    "day, sp, utc",
    [
        pytest.param(date(2025, 1, 15), 1, "2025-01-15 00:00", id="winter: SP1 is midnight UTC"),
        pytest.param(date(2025, 6, 18), 1, "2025-06-17 23:00", id="summer: SP1 starts the evening before in UTC"),
        pytest.param(date(2025, 3, 30), 3, "2025-03-30 01:00", id="clocks forward: SP3 is 02:00 BST"),
        pytest.param(date(2025, 3, 30), 46, "2025-03-30 22:30", id="clocks forward: SP46 is 23:30 BST"),
        pytest.param(date(2025, 10, 26), 3, "2025-10-26 00:00", id="clocks back: SP3 is the first 01:00"),
        pytest.param(date(2025, 10, 26), 5, "2025-10-26 01:00", id="clocks back: SP5 is the second 01:00"),
        pytest.param(date(2025, 10, 26), 50, "2025-10-26 23:30", id="clocks back: SP50 is 23:30 GMT"),
    ],
)
def test_sp_start_utc(day, sp, utc):
    assert pd.Timestamp(sp_start_utc(day, sp)) == pd.Timestamp(utc, tz="UTC")


def test_sp_start_utc_refuses_a_period_that_does_not_exist():
    with pytest.raises(ValueError):
        sp_start_utc(date(2025, 3, 30), 47)


def test_parse_indo_keeps_latest_publication_and_drops_nulls():
    rows = [
        {"settlementDate": "2025-01-15", "settlementPeriod": 1, "initialDemandOutturn": 25000,
         "publishTime": "2025-01-15T00:45:00Z", "startTime": "x", "initialTransmissionSystemDemandOutturn": 1},
        {"settlementDate": "2025-01-15", "settlementPeriod": 1, "initialDemandOutturn": 25100,
         "publishTime": "2025-01-15T01:15:00Z", "startTime": "x", "initialTransmissionSystemDemandOutturn": 1},
        {"settlementDate": "2025-01-15", "settlementPeriod": 2, "initialDemandOutturn": None,
         "publishTime": "2025-01-15T01:15:00Z", "startTime": "x", "initialTransmissionSystemDemandOutturn": 1},
    ]
    df = parse_indo(rows)
    assert df["settlement_period"].tolist() == [1]
    assert df["demand_mw"].tolist() == [25100.0]
    assert str(df["publish_time"].dt.tz) == "UTC"


def test_parse_ndf_keeps_every_vintage():
    rows = [
        {"dataset": "NDF", "settlementDate": "2025-01-15", "settlementPeriod": 1, "demand": 26000,
         "publishTime": "2025-01-14T09:05:00Z", "startTime": "x", "boundary": "N"},
        {"dataset": "NDF", "settlementDate": "2025-01-15", "settlementPeriod": 1, "demand": 26200,
         "publishTime": "2025-01-14T12:05:00Z", "startTime": "x", "boundary": "N"},
        {"dataset": "NDF", "settlementDate": "2025-01-15", "settlementPeriod": 1, "demand": 999,
         "publishTime": "2025-01-14T12:05:00Z", "startTime": "x", "boundary": "B1"},
    ]
    df = parse_ndf(rows)
    assert df["demand_mw"].tolist() == [26000.0, 26200.0]


def test_missing_periods_reports_gaps_but_accepts_a_complete_46_period_day():
    days = {date(2025, 3, 29): 48, date(2025, 3, 30): 46}
    rows = [dict(settlement_date=pd.Timestamp(d), settlement_period=sp) for d, n in days.items() for sp in range(1, n + 1)]
    df = pd.DataFrame(rows)
    df = df[~((df["settlement_date"] == pd.Timestamp(2025, 3, 29)) & (df["settlement_period"] == 7))]
    assert missing_periods(df, date(2025, 3, 29), date(2025, 3, 30)) == {date(2025, 3, 29): ([7], [])}


class FakeResponse:
    """Stands in for requests' response, so the test needs no network."""
    def __init__(self, rows):
        self.rows = rows

    def raise_for_status(self):
        pass

    def json(self):
        return {"data": self.rows}


def test_fetch_mid_prices_returns_nan_where_nothing_traded(monkeypatch):
    rows = [dict(settlementDate="2025-06-18", settlementPeriod=sp, price=50.0 + sp, volume=100.0)
            for sp in range(1, 49)]
    rows[9].update(price=0.0, volume=0.0)  # SP 10: no trades, so Elexon shows a £0 placeholder
    monkeypatch.setattr(elexon.requests, "get", lambda *args, **kwargs: FakeResponse(rows))
    with pytest.warns(UserWarning, match="no traded volume"):
        prices = elexon.fetch_mid_prices(date(2025, 6, 18))
    assert math.isnan(prices[10])
    assert prices[11] == 61.0
    assert sum(math.isnan(p) for p in prices.values()) == 1