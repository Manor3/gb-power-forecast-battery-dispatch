"""Fetch GB data from Elexon's Insights API (free, no key).

Session 1: Market Index Data (MID) prices.
Session 2: INDO demand outturn, and NESO's NDF day-ahead demand forecast with
every vintage's publish time.

MID is an index of short-term exchange trades (APX), not the day-ahead
auction price: say so wherever you report results.
"""
from datetime import datetime, time, timedelta, timezone
import warnings
from zoneinfo import ZoneInfo

import pandas as pd
import requests

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
MID_URL = f"{BASE_URL}/balancing/pricing/market-index"
INDO_URL = f"{BASE_URL}/demand/outturn"
NDF_URL = f"{BASE_URL}/datasets/NDF"
LONDON = ZoneInfo("Europe/London")

INDO_COLUMNS = ["settlement_date", "settlement_period", "demand_mw", "publish_time"]
NDF_COLUMNS = ["settlement_date", "settlement_period", "publish_time", "demand_mw"]

# NESO publishes the day-ahead NDF at 09:00-09:15 and 12:00-12:15 local time. One request
# per target day D, covering publications on D-1 between these UTC times, holds both.
NDF_WINDOW_UTC = (time(0, 0), time(13, 0))  # the endpoint allows at most 1 day per request


def expected_periods(day):
    """Settlement periods in a GB settlement day: 46 when clocks go forward, 50 when they go back."""
    last_sunday = day.weekday() == 6 and (day + timedelta(days=7)).month != day.month
    if last_sunday and day.month == 3:
        return 46
    if last_sunday and day.month == 10:
        return 50
    return 48


def sp_start_utc(day, sp):
    """UTC start of settlement period sp on settlement date day.

    SP 1 starts at local midnight and every SP is 30 minutes of real time, so on
    the 46- and 50-period days SP numbers stop matching clock time.
    """
    if not 1 <= sp <= expected_periods(day):
        raise ValueError(f"{day} has no settlement period {sp}")
    midnight = datetime(day.year, day.month, day.day, tzinfo=LONDON)
    return midnight.astimezone(timezone.utc) + timedelta(minutes=30 * (sp - 1))


# --- Session 1: prices ---

def fetch_mid_prices(day, provider="APXMIDP"):
    """Market Index prices for one settlement date, as {settlement period: £/MWh}.

    Raises ValueError if any settlement period is missing or duplicated.
    """
    stamp = f"{day.isoformat()}T00:00Z"
    params = {
        "from": stamp,
        "to": stamp,
        "settlementPeriodFrom": 1,
        "settlementPeriodTo": 50,
        "dataProviders": provider,
        "format": "json",
    }
    response = requests.get(MID_URL, params=params, timeout=30)
    response.raise_for_status()
    rows = [r for r in response.json()["data"] if r["settlementDate"] == day.isoformat()]

    n = expected_periods(day)
    got = sorted(r["settlementPeriod"] for r in rows)
    if got != list(range(1, n + 1)):
        missing = sorted(set(range(1, n + 1)) - set(got))
        raise ValueError(f"{day}: expected {n} settlement periods, got {len(got)}; missing {missing}")

    zero_volume = [r["settlementPeriod"] for r in rows if not r["volume"]]
    if zero_volume:
        warnings.warn(f"{day}: no traded volume in periods {zero_volume}; those prices may be placeholders")

    return {r["settlementPeriod"]: r["price"] if r["volume"] else float("nan") for r in rows}


# --- Session 2: demand outturn (INDO) ---

def _empty(columns):
    return _typed(pd.DataFrame({c: [] for c in columns}), columns)


def _typed(df, columns):
    """Shared clean-up: settlement_date as midnight, SP as int, publish time in UTC."""
    df = df.copy()
    df["settlement_date"] = pd.to_datetime(df["settlement_date"]).astype("datetime64[ns]")
    df["settlement_period"] = df["settlement_period"].astype(int)
    df["publish_time"] = pd.to_datetime(df["publish_time"], utc=True).astype("datetime64[ns, UTC]")
    df["demand_mw"] = df["demand_mw"].astype(float)
    return df[columns]


def parse_indo(rows):
    """Rows from /demand/outturn -> one row per settlement period.

    Drops rows with no demand value (the completeness check then reports them as
    missing) and keeps the latest publication if a period appears twice.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return _empty(INDO_COLUMNS)
    df = df.rename(columns={
        "settlementDate": "settlement_date",
        "settlementPeriod": "settlement_period",
        "initialDemandOutturn": "demand_mw",
        "publishTime": "publish_time",
    })
    df = df[df["demand_mw"].notna()]
    if df.empty:
        return _empty(INDO_COLUMNS)
    df = _typed(df, INDO_COLUMNS)
    df = df.sort_values("publish_time", na_position="first")
    df = df.drop_duplicates(["settlement_date", "settlement_period"], keep="last")
    return df.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)


def fetch_indo(start, end, chunk_days=7):
    """INDO for settlement dates start..end inclusive. One request per chunk of days."""
    frames = []
    day = start
    while day <= end:
        stop = min(day + timedelta(days=chunk_days - 1), end)
        params = {"settlementDateFrom": day.isoformat(), "settlementDateTo": stop.isoformat(), "format": "json"}
        response = requests.get(INDO_URL, params=params, timeout=60)
        response.raise_for_status()
        frame = parse_indo(response.json()["data"])
        if not frame.empty:
            frames.append(frame)
        day = stop + timedelta(days=1)
    if not frames:
        return _empty(INDO_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    inside = df["settlement_date"].between(pd.Timestamp(start), pd.Timestamp(end))
    return df[inside].reset_index(drop=True)


def missing_periods(df, start, end):
    """{day: (missing SPs, unexpected SPs)} for every incomplete settlement date in start..end."""
    have = df.groupby("settlement_date")["settlement_period"].agg(set).to_dict()
    problems = {}
    for ts in pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D"):
        expected = set(range(1, expected_periods(ts.date()) + 1))
        got = have.get(ts, set())
        if got != expected:
            problems[ts.date()] = (sorted(expected - got), sorted(got - expected))
    return problems


# --- Session 2: NESO's day-ahead national demand forecast (NDF) ---

def parse_ndf(rows):
    """Rows from /datasets/NDF -> [settlement_date, settlement_period, publish_time, demand_mw].

    One row per (target SP, vintage): the same SP appears once per publication.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return _empty(NDF_COLUMNS)
    if "boundary" in df.columns:  # NDF is national only; be safe if zonal rows ever appear
        df = df[df["boundary"].isna() | (df["boundary"] == "N")]
    df = df.rename(columns={
        "settlementDate": "settlement_date",
        "settlementPeriod": "settlement_period",
        "publishTime": "publish_time",
        "demand": "demand_mw",
    })
    df = df[df["demand_mw"].notna() & df["publish_time"].notna()]
    if df.empty:
        return _empty(NDF_COLUMNS)
    df = _typed(df, NDF_COLUMNS)
    df = df.drop_duplicates(["settlement_date", "settlement_period", "publish_time"])
    return df.sort_values(["settlement_date", "settlement_period", "publish_time"]).reset_index(drop=True)


def fetch_ndf_vintages(target_day):
    """Every NDF vintage for settlement date target_day published on the day before, in NDF_WINDOW_UTC."""
    publish_day = target_day - timedelta(days=1)
    t_from = datetime.combine(publish_day, NDF_WINDOW_UTC[0], tzinfo=timezone.utc)
    t_to = datetime.combine(publish_day, NDF_WINDOW_UTC[1], tzinfo=timezone.utc)
    params = {
        "publishDateTimeFrom": t_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "publishDateTimeTo": t_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "format": "json",
    }
    response = requests.get(NDF_URL, params=params, timeout=60)
    response.raise_for_status()
    df = parse_ndf(response.json()["data"])
    return df[df["settlement_date"] == pd.Timestamp(target_day)].reset_index(drop=True)
