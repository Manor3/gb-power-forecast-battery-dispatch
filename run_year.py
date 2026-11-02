"""Step 8: a full year of perfect-foresight dispatch, with a cycle-cap sweep.

The first run fetches every day from Elexon and caches it in data/.
Later runs read the cache. Days that fail are logged, never silently dropped.
"""
from datetime import date, timedelta
from pathlib import Path
import math
import time
import warnings

import pandas as pd
import requests

from battery import build_model, check_solution, cycles_used, run
from elexon import expected_periods, fetch_mid_prices

YEAR = 2025
# 1 MW / 2 MWh battery, 88% round-trip efficiency split evenly, starts and ends half full
PARAMS = dict(dt=0.5, P_max=1.0, E_max=2.0, eta_c=math.sqrt(0.88), eta_d=math.sqrt(0.88), e0=1.0)
CAPS = [None, 2, 1.5, 1]  # MIP cycle caps to sweep; None = uncapped

DATA = Path("data")
PRICES_CSV = DATA / f"mid_prices_{YEAR}.csv"
FETCH_LOG_CSV = DATA / f"fetch_log_{YEAR}.csv"
RESULTS_CSV = DATA / f"results_{YEAR}.csv"
LOG_CSV = DATA / f"log_{YEAR}.csv"
LOG_COLUMNS = ["date", "stage", "message"]


def days_in(year):
    day = date(year, 1, 1)
    while day.year == year:
        yield day
        day += timedelta(days=1)


def case_name(cap):
    return "mip_uncapped" if cap is None else f"mip_cap_{cap:g}"


def fetch_with_retry(day, attempts=3, wait=2.0):
    """Retry network errors only: a missing settlement period won't fix itself."""
    for attempt in range(1, attempts + 1):
        try:
            return fetch_mid_prices(day)
        except requests.RequestException:
            if attempt == attempts:
                raise
            time.sleep(wait * attempt)


def fetch_year(year):
    """Fetch every day from Elexon, then save prices and the fetch log to data/."""
    DATA.mkdir(exist_ok=True)
    rows, fetch_log = [], []
    for day in days_in(year):
        if day.day == 1:
            print(f"fetching {day:%B %Y}...")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                day_prices = fetch_with_retry(day)
        except (requests.RequestException, ValueError, KeyError) as err:
            fetch_log.append(dict(date=day.isoformat(), stage="fetch", message=str(err)))
            continue
        for w in caught:
            fetch_log.append(dict(date=day.isoformat(), stage="fetch warning", message=str(w.message)))
        rows += [dict(settlement_date=day.isoformat(), settlement_period=p, price=v)
                 for p, v in day_prices.items()]
    pd.DataFrame(rows).to_csv(PRICES_CSV, index=False)
    pd.DataFrame(fetch_log, columns=LOG_COLUMNS).to_csv(FETCH_LOG_CSV, index=False)


def load_prices(year, log):
    """{day: {period: price}} from the cache, fetching first if there's no cache yet."""
    if not PRICES_CSV.exists():
        fetch_year(year)
    log += pd.read_csv(FETCH_LOG_CSV).to_dict("records")

    df = pd.read_csv(PRICES_CSV)
    prices = {}
    for day_str, g in df.groupby("settlement_date"):
        day = date.fromisoformat(day_str)
        day_prices = {int(p): float(v) for p, v in zip(g["settlement_period"], g["price"])}
        if sorted(day_prices) != list(range(1, expected_periods(day) + 1)):
            log.append(dict(date=day_str, stage="cache", message="wrong settlement periods in cache"))
            continue
        prices[day] = day_prices
    return prices


def solve_year(prices, log):
    """One row per day: revenue and cycles for the LP and each MIP cap."""
    cases = [("lp", False, None)] + [(case_name(cap), True, cap) for cap in CAPS]
    rows = []
    for day in sorted(prices):
        if day.day == 1:
            print(f"solving {day:%B}...")
        p = prices[day]
        traded = [v for v in p.values() if not math.isnan(v)]  # min() and max() mishandle NaN
        row = dict(date=day, periods=len(p), idle_periods=len(p) - len(traded),
           min_price=min(traded), max_price=max(traded))
        for name, binary, cap in cases:
            try:
                m = build_model(prices=p, binary=binary, n_cycles=cap, **PARAMS)
                revenue = run(m)
                check_solution(m, binary=binary, n_cycles=cap, **PARAMS)
            except (RuntimeError, ValueError) as err:
                log.append(dict(date=day.isoformat(), stage=name, message=str(err)))
                row[name] = float("nan")
                continue
            row[name] = revenue
            row[f"cycles_{name}"] = cycles_used(m, PARAMS["eta_d"], PARAMS["dt"], PARAMS["E_max"])
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(results, log):
    n_days = sum(1 for _ in days_in(YEAR))
    print(f"\n{YEAR}: {len(results)} of {n_days} days solved; {len(log)} log entries (see {LOG_CSV})")
    print("1 MW / 2 MWh, 88% round trip, perfect foresight on APX Market Index prices\n")

    names = ["lp"] + [case_name(cap) for cap in CAPS]
    table = pd.DataFrame(
        {
            "£/MW/yr": [results[n].sum() for n in names],
            "avg cycles/day": [results.get(f"cycles_{n}", pd.Series(dtype=float)).mean() for n in names],
            "days solved": [int(results[n].notna().sum()) for n in names],
        },
        index=names,
    )
    print(table.round(2).to_string())

    gap = results["lp"] - results["mip_uncapped"]
    print(f"\nLP - MIP (uncapped): £{gap.sum():,.2f}/MW/yr; days with a gap above £0.01: {(gap > 0.01).sum()}; "
          f"days with a negative price: {(results['min_price'] < 0).sum()}")

    print("\nValue of extra cycling (tightest cap first):")
    order = sorted([c for c in CAPS if c is not None]) + [None]
    for tighter, looser in zip(order, order[1:]):
        a, b = case_name(tighter), case_name(looser)
        d_revenue = results[b].sum() - results[a].sum()
        d_cycles = results.get(f"cycles_{b}", pd.Series(dtype=float)).sum() - results.get(f"cycles_{a}", pd.Series(dtype=float)).sum()
        per_cycle = d_revenue / d_cycles if d_cycles > 1e-9 else float("nan")
        print(f"  {a} -> {b}: +£{d_revenue:,.0f} for +{d_cycles:,.0f} cycles = £{per_cycle:,.2f} per extra cycle")


if __name__ == "__main__":
    log = []
    prices = load_prices(YEAR, log)
    results = solve_year(prices, log)
    results.to_csv(RESULTS_CSV, index=False)
    pd.DataFrame(log, columns=LOG_COLUMNS).to_csv(LOG_CSV, index=False)
    summarise(results, log)
