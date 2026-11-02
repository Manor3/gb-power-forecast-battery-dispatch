"""Session 2: a GB day-ahead demand forecast, scored point-in-time.

The formulation, as this code implements it:
1. When: at τ = 09:30 on D-1, forecast every settlement period of D.
2. What: national demand as the grid sees it (INDO), in MW.
3. Allowed: only data PUBLISHED by τ: demand up to D-2 (from v2, also D-1 until 08:00), weather
   forecast 48 hours before each half-hour, and the calendar. Reanalysis weather only for
   training rows and D-7.
4. Forecasters: the naive (same local half-hour on D-7) is the floor; NESO's last NDF
   published by τ is the benchmark; the GBM never sees NESO's forecast.
5. Score: MAE, RMSE, bias, MASE (scale fixed on the training years) and skill vs the naive,
   by half-hour, day type and month, on the common sample; Diebold-Mariano on daily losses.
6. Protocol: daily issue, monthly refit, expanding window from 2022. Settings frozen on
   2024, 2025 scored once. The database enforces both.

How to run:
- YEAR = 2024 while you iterate. New settings need a new name in gbm.FEATURE_SETS.
- Running a version that already ran re-scores its stored forecasts: nothing is overwritten.
- When one version is final, set YEAR = 2025 and run once. Never edit settings after that.
The first run fetches everything and caches it in data/; later runs read the cache.
"""
from datetime import date, timedelta
from pathlib import Path
import time

import pandas as pd
import requests

import backtest
import elexon
import evaluate
import gbm
import gbtime
import pit
import store
import weather

YEAR = 2025                     # 2024 = validation (iterate); 2025 = the test (once)
DEV_YEAR, TEST_YEAR = 2024, 2025
MODEL_VERSION = "v1"            # a key of gbm.FEATURE_SETS; new settings need a new name
WEATHER_DIAGNOSTIC = True       # also score the GBM fed the REAL weather: what forecast error costs
TRAIN_START = date(2022, 1, 1)
HISTORY_START = TRAIN_START - timedelta(days=31)  # the naive and the lags reach back up to 14 days
MIN_NESO_COVERAGE = 0.95        # stop if NESO has nothing by τ on more days than this allows

DATA = Path("data")
DB_PATH = DATA / "session2.db"
RESULTS = DATA / "results"
LOG_COLUMNS = ["date", "stage", "message"]


# --- fetching and caching ---

def with_retry(fn, *args, attempts=3, wait=2.0):
    """Retry network errors only: a missing settlement period won't fix itself."""
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args)
        except requests.RequestException:
            if attempt == attempts:
                raise
            time.sleep(wait * attempt)


def months(start, end):
    """(first, last) day of each whole calendar month touching start..end. Caches hold whole months."""
    m, last = pd.Period(start, "M"), pd.Period(end, "M")
    while m <= last:
        yield m.start_time.date(), m.end_time.date()
        m += 1


def settled(month_end, delay_days=7):
    """Only cache a month once nothing in it can still change (ERA5 arrives about 5 days late)."""
    return month_end < date.today() - timedelta(days=delay_days)


def read_cached(path, kind):
    df = pd.read_csv(path)
    if kind == "weather":
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True).astype("datetime64[ns, UTC]")
        return df
    df["settlement_date"] = pd.to_datetime(df["settlement_date"]).astype("datetime64[ns]")
    df["publish_time"] = pd.to_datetime(df["publish_time"], utc=True).astype("datetime64[ns, UTC]")
    return df


def save(df, path, ok):
    if ok:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


def load_indo(start, end, log):
    frames = []
    for m0, m1 in months(start, end):
        path = DATA / "indo" / f"{m0:%Y-%m}.csv"
        if path.exists():
            frames.append(read_cached(path, "indo"))
            continue
        print(f"  INDO {m0:%Y-%m}")
        df = with_retry(elexon.fetch_indo, m0, m1)
        save(df, path, settled(m1))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["settlement_date"].between(pd.Timestamp(start), pd.Timestamp(end))].reset_index(drop=True)
    for day, (missing, extra) in elexon.missing_periods(df, start, end).items():
        note = f"missing SPs {missing}" + (f"; unexpected SPs {extra}" if extra else "")
        log.append(dict(date=day.isoformat(), stage="indo", message=note))
    return df


def load_weather(kind, start, end, log):
    """Half-hourly national weather for every SP of start..end, with each value's info_time."""
    grid = gbtime.sp_grid(start, end)
    first = grid["start_utc"].min().date()  # in summer, SP 1 starts at 23:00 UTC the day before
    last = (grid["start_utc"].max() + pd.Timedelta(hours=2)).date()  # the last SP interpolates into the next hour
    frames = []
    for m0, m1 in months(first, last):
        path = DATA / "weather" / f"{kind}_{m0:%Y-%m}.csv"
        if path.exists():
            frames.append(read_cached(path, "weather"))
            continue
        print(f"  weather {kind} {m0:%Y-%m}")
        df = with_retry(weather.fetch_sites, kind, m0, m1)
        save(df, path, settled(m1))
        frames.append(df)
    hourly = weather.national(pd.concat(frames, ignore_index=True))
    hh = weather.to_half_hourly(hourly, grid["start_utc"])
    hh["info_time"] = weather.info_time(kind, hh.index)
    gaps = hh[gbm.WEATHER_VARS].isna().any(axis=1)
    if gaps.any():
        days = sorted(set(grid.loc[gaps.to_numpy(), "settlement_date"].dt.date))
        log.append(dict(date="", stage=f"weather {kind}",
                        message=f"{int(gaps.sum())} SPs on {len(days)} days without data, from {days[0]} to {days[-1]}"))
    return hh


def load_ndf(start, end, log):
    frames = []
    for m0, m1 in months(start, end):
        path = DATA / "ndf" / f"{m0:%Y-%m}.csv"
        if path.exists():
            frames.append(read_cached(path, "ndf"))
            continue
        print(f"  NDF {m0:%Y-%m}")
        month, failed = [], False
        for day in pd.date_range(m0, m1, freq="D").date:
            try:
                month.append(with_retry(elexon.fetch_ndf_vintages, day))
            except (requests.RequestException, ValueError, KeyError) as err:
                failed = True
                log.append(dict(date=day.isoformat(), stage="ndf fetch", message=str(err)))
        month = [f for f in month if len(f)]
        df = pd.concat(month, ignore_index=True) if month else elexon.parse_ndf([])
        save(df, path, settled(m1) and not failed)  # a month with a failed day is fetched again next run
        frames.append(df)
    frames = [f for f in frames if len(f)]
    if not frames:
        return elexon.parse_ndf([])
    df = pd.concat(frames, ignore_index=True)
    return df[df["settlement_date"].between(pd.Timestamp(start), pd.Timestamp(end))].reset_index(drop=True)


# --- the run ---

def config_for(version):
    """Everything that defines the forecaster. 2025 only accepts a config already run on 2024."""
    return dict(
        version=version, features=gbm.FEATURE_SETS[version], params=gbm.PARAMS,
        issue_time=gbtime.ISSUE_TIME.isoformat(), train_start=TRAIN_START.isoformat(), refit="monthly",
        sites=weather.SITES, weighting=weather.WEIGHTING, reanalysis=weather.REANALYSIS_MODEL,
        forecast_model=weather.FORECAST_MODEL, forecast_lead_days=weather.FORECAST_LEAD_DAYS,
    )


def check_neso_timing(ndf, taus):
    """Which NESO forecast gets graded (its last one published by τ), and how often one exists."""
    graded = pit.pick_vintage(ndf, taus)
    if graded.empty:
        raise ValueError("no NESO forecast published by τ on any day: check the 'ndf fetch' entries in the log")
    share = graded["settlement_date"].nunique() / len(taus)
    published = graded["publish_time"].dt.tz_convert(gbtime.TZ).dt.strftime("%H:%M")
    top = ", ".join(f"{t} ({s:.0%})" for t, s in published.value_counts(normalize=True).head(3).items())
    print(f"NESO is graded on its last forecast published by {gbtime.ISSUE_TIME:%H:%M} on D-1, "
          f"most often published at {top} local. One was out on {share:.1%} of days.")
    if share < MIN_NESO_COVERAGE:
        raise ValueError(f"NESO had a forecast out by τ on only {share:.1%} of days: "
                         f"check the 'ndf fetch' entries in the log before trusting any score.")


def forecast_year(con, model, days, log):
    feats = gbm.FEATURE_SETS[MODEL_VERSION]
    uses_weather = any(f in gbm.WEATHER for f in feats)
    config = config_for(MODEL_VERSION)
    store.check_protocol(con, model, YEAR, config, DEV_YEAR, TEST_YEAR)  # before any work

    print("loading data (first run fetches it)...")
    outturn = load_indo(HISTORY_START, days[-1], log)
    wx_act = load_weather("reanalysis", HISTORY_START, days[-1], log) if uses_weather else None
    wx_fc = load_weather("forecast", days[0], days[-1], log) if uses_weather else None
    ndf = load_ndf(days[0], days[-1], log)
    check_neso_timing(ndf, gbtime.issue_times(days))

    print(f"backtesting {model} on {YEAR}: daily issue at τ, monthly refit...")
    forecasts, panel, bt_log = backtest.run_backtest(
        outturn, wx_act, wx_fc, ndf, days, TRAIN_START, MODEL_VERSION, diagnostic=WEATHER_DIAGNOSTIC)
    log += bt_log
    store.upsert_outturn(con, outturn)
    store.save_run(con, model, YEAR, config, forecasts)
    return panel


def score_year(con, model, panel, log):
    year_start, year_end = date(YEAR, 1, 1), date(YEAR, 12, 31)
    contest = ["naive", "neso", model]
    diag = f"{model}_actualwx"
    table = store.read_scored(con, contest + [diag], year_start, year_end)
    has_diag = table[diag].notna().any()
    table = evaluate.common_sample(table, contest + ([diag] if has_diag else []), log)
    table = evaluate.add_groups(table)
    shown = contest + ([diag] if has_diag else [])
    scale = evaluate.mase_scale(panel, TRAIN_START, year_start)
    label = "validation: iterate freely" if YEAR == DEV_YEAR else "TEST: scored once"

    print(f"\n{YEAR} ({label}): {table['settlement_date'].nunique()} days, {len(table):,} SPs on the common sample")
    print(f"issue time {gbtime.ISSUE_TIME:%H:%M} on D-1; INDO, MW; MASE scale = naive MAE on "
          f"{TRAIN_START}..{year_start - timedelta(days=1)} = {scale:,.0f} MW")
    if has_diag:
        print(f"{diag} is fed the real weather: a diagnostic, never a contestant\n")

    tables = {
        "summary": evaluate.summary(table, shown, scale),
        "by_half_hour": evaluate.mae_by(table, shown, "half_hour"),
        "by_day_type": evaluate.mae_by(table, shown, "day_type"),
        "by_month": evaluate.mae_by(table, shown, "month"),
        "diebold_mariano": evaluate.dm_table(table, [(model, "naive"), (model, "neso"), ("neso", "naive")]),
    }
    rolling = store.rolling_mae(con)
    rolling = rolling[rolling["model"].isin(shown) & rolling["day"].str.startswith(str(YEAR))]
    tables["rolling_mae_28d"] = rolling.pivot(index="day", columns="model", values="rolling_mae")

    print(tables["summary"].round({"MAE": 0, "RMSE": 0, "bias": 0, "MASE": 3, "skill": 3, "RMSE/MAE": 2,
                                   "gap_closed": 3}).to_string())
    print("\nDiebold-Mariano on daily mean loss (abs: MAE-type, in MW; sq: RMSE-type, in MW²). "
          "Negative mean_diff favours the first named:")
    print(tables["diebold_mariano"].round({"mean_diff": 1, "dm_stat": 2, "p_value": 4}).to_string(index=False))
    for key in ("by_day_type", "by_month", "by_half_hour"):
        print(f"\nMAE {key.replace('_', ' ')} (MW):")
        print(tables[key].round(0).to_string())
    month_ends = tables["rolling_mae_28d"].groupby(pd.to_datetime(tables["rolling_mae_28d"].index).to_period("M")).tail(1)
    print("\n28-day rolling MAE at each month end (SQL window function, RANGE over days):")
    print(month_ends.round(0).to_string())

    s = tables["summary"]
    if has_diag:
        cost = s.loc[model, "MAE"] - s.loc[diag, "MAE"]
        print(f"\nWeather diagnostic: real weather would cut {model}'s MAE from {s.loc[model, 'MAE']:,.0f} "
              f"to {s.loc[diag, 'MAE']:,.0f} MW, so weather-forecast error costs {cost:,.0f} MW of MAE.")
    versions = store.models_run(con, YEAR)
    if len(versions) > 1:  # every version on the same half-hours, so they can be ranked
        field = ["naive", "neso", *versions]
        common = evaluate.common_sample(store.read_scored(con, field, year_start, year_end), field, [])
        tables["versions"] = evaluate.summary(common, field, scale)[["MAE", "RMSE", "bias", "skill", "gap_closed"]]
        tables["versions_dm"] = evaluate.dm_table(common, [(model, v) for v in versions if v != model])
        print(f"\nEvery version run on {YEAR}, on the same {len(common):,} half-hours:")
        print(tables["versions"].round({"MAE": 0, "RMSE": 0, "bias": 0, "skill": 3, "gap_closed": 3}).to_string())
        print(f"\nDiebold-Mariano, {model} against each other version:")
        print(tables["versions_dm"].round({"mean_diff": 1, "dm_stat": 2, "p_value": 4}).to_string(index=False))

    dm = tables["diebold_mariano"].set_index(["pair", "loss"])
    print(f"\nInterview line: in {YEAR}, my GBM's MAE was {s.loc[model, 'MAE']:,.0f} MW against "
          f"{s.loc['naive', 'MAE']:,.0f} for the naive and {s.loc['neso', 'MAE']:,.0f} for NESO, closing "
          f"{s.loc[model, 'gap_closed']:.0%} of the gap; GBM vs NESO DM p = {dm.loc[(f'{model} vs neso', 'abs'), 'p_value']:.3f}.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    for key, t in tables.items():
        t.to_csv(RESULTS / f"{YEAR}_{model}_{key}.csv")
    table.to_csv(RESULTS / f"{YEAR}_{model}_scored_sps.csv", index=False)


def year_days(year):
    return list(pd.date_range(date(year, 1, 1), date(year, 12, 31), freq="D").date)


def main():
    if MODEL_VERSION not in gbm.FEATURE_SETS:
        raise ValueError(f"MODEL_VERSION = {MODEL_VERSION!r} isn't defined; gbm.FEATURE_SETS has "
                         f"{', '.join(gbm.FEATURE_SETS)}. A version is a feature list: add the new one there first.")
    DATA.mkdir(parents=True, exist_ok=True)
    log = []
    model = f"gbm_{MODEL_VERSION}"
    days = year_days(YEAR)
    con = store.connect(DB_PATH)
    stored = store.existing_run(con, model, YEAR)
    log_path = DATA / (f"log_{YEAR}_{model}.csv" if stored is None else f"log_{YEAR}_{model}_rescore.csv")
    try:
        if stored is None:
            panel = forecast_year(con, model, days, log)
        else:
            print(f"{model} already ran on {YEAR}. Stored forecasts are never overwritten, so this run only "
                  f"re-scores them.\nTo try new settings, add them to gbm.FEATURE_SETS under a new name and "
                  f"set MODEL_VERSION to it.")
            if stored != store.config_json(config_for(MODEL_VERSION)):
                print(f"Note: your settings have changed since {model} ran. These scores are for the settings it ran with.")
            panel = backtest.build_panel(load_indo(HISTORY_START, days[-1], log), HISTORY_START, days[-1])
        score_year(con, model, panel, log)
    finally:
        con.close()
        pd.DataFrame(log, columns=LOG_COLUMNS).to_csv(log_path, index=False)
        print(f"\n{len(log)} log entries in {log_path}")


if __name__ == "__main__":
    main()
