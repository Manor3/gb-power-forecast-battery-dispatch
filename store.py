"""SQLite store: append-only forecasts, the outturn, and a register of runs.

The database enforces the protocol, so it doesn't rest on good intentions:
1. A forecast, once written, is never changed (append-only).
2. Every GBM attempt has its own name, so the record of trials is complete.
3. The test year is scored once, by a configuration already run on the validation year.
"""
import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    model             TEXT    NOT NULL,
    issue_time        TEXT    NOT NULL,  -- τ, UTC
    settlement_date   TEXT    NOT NULL,
    settlement_period INTEGER NOT NULL,
    forecast_mw       REAL    NOT NULL,
    source_time       TEXT,              -- NESO only: when the vintage used was published
    PRIMARY KEY (model, issue_time, settlement_date, settlement_period)
);
CREATE TABLE IF NOT EXISTS outturn (
    settlement_date   TEXT    NOT NULL,
    settlement_period INTEGER NOT NULL,
    demand_mw         REAL    NOT NULL,
    publish_time      TEXT,
    PRIMARY KEY (settlement_date, settlement_period)
);
CREATE TABLE IF NOT EXISTS runs (
    model       TEXT    NOT NULL,
    year        INTEGER NOT NULL,
    config      TEXT    NOT NULL,
    created_utc TEXT    NOT NULL,
    PRIMARY KEY (model, year)
);
"""

BENCHMARKS = ("naive", "neso")  # deterministic: rewriting them is skipped, not an error


def connect(path):
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def _iso(ts):
    return None if pd.isna(ts) else pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def config_json(config):
    return json.dumps(config, sort_keys=True, default=str)


def existing_run(con, model, year):
    """The settings model ran with on year (as stored JSON), or None if it hasn't run."""
    row = con.execute("SELECT config FROM runs WHERE model = ? AND year = ?", (model, year)).fetchone()
    return None if row is None else row[0]


def models_run(con, year):
    """Every GBM version with a run on year, in name order."""
    return [r[0] for r in con.execute("SELECT model FROM runs WHERE year = ? ORDER BY model", (year,))]


def check_protocol(con, model, year, config, dev_year, test_year):
    """Raise BEFORE any work if a NEW run of model on year would break the protocol."""
    runs = dict(con.execute("SELECT year, config FROM runs WHERE model = ?", (model,)).fetchall())
    if year in runs:
        raise ValueError(f"{model} already ran on {year}, and stored forecasts are never overwritten.")
    if year == test_year:
        if runs.get(dev_year) != config_json(config):
            raise ValueError(f"{model} has no {dev_year} run with exactly this configuration. "
                             f"{test_year} only scores settings frozen on {dev_year}.")
        taken = con.execute("SELECT model FROM runs WHERE year = ?", (test_year,)).fetchone()
        if taken:
            raise ValueError(f"{test_year} was already scored by {taken[0]}: the test year is looked at once.")


def _records(df):
    return [
        (r.model, _iso(r.issue_time), pd.Timestamp(r.settlement_date).date().isoformat(),
         int(r.settlement_period), float(r.forecast_mw), _iso(r.source_time))
        for r in df.itertuples(index=False)
    ]


def save_run(con, model, year, config, forecasts):
    """Register the run and append its forecasts in ONE transaction: all or nothing."""
    bench = forecasts["model"].isin(BENCHMARKS)
    with con:
        con.execute("INSERT INTO runs VALUES (?, ?, ?, ?)",
                    (model, year, config_json(config), datetime.now(timezone.utc).isoformat(timespec="seconds")))
        con.executemany("INSERT OR IGNORE INTO forecasts VALUES (?, ?, ?, ?, ?, ?)", _records(forecasts[bench]))
        con.executemany("INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?)", _records(forecasts[~bench]))


def upsert_outturn(con, outturn):
    """The outturn is data, not a forecast: a corrected value may replace an old one."""
    rows = [(pd.Timestamp(r.settlement_date).date().isoformat(), int(r.settlement_period), float(r.demand_mw),
             _iso(r.publish_time)) for r in outturn.itertuples(index=False)]
    with con:
        con.executemany("INSERT OR REPLACE INTO outturn VALUES (?, ?, ?, ?)", rows)


def read_scored(con, models, start, end):
    """One row per SP in start..end: the outturn ('actual') and one column per model."""
    span = (start.isoformat(), end.isoformat())
    marks = ",".join("?" * len(models))
    fc = pd.read_sql_query(
        f"SELECT model, settlement_date, settlement_period, forecast_mw FROM forecasts "
        f"WHERE model IN ({marks}) AND settlement_date BETWEEN ? AND ?", con, params=[*models, *span])
    ot = pd.read_sql_query(
        "SELECT settlement_date, settlement_period, demand_mw AS actual FROM outturn "
        "WHERE settlement_date BETWEEN ? AND ?", con, params=list(span))
    wide = fc.pivot_table(index=["settlement_date", "settlement_period"], columns="model", values="forecast_mw")
    wide = wide.reindex(columns=list(models))
    table = ot.set_index(["settlement_date", "settlement_period"]).join(wide, how="outer").reset_index()
    table["settlement_date"] = pd.to_datetime(table["settlement_date"])
    table.columns.name = None
    return table


def rolling_mae(con, window_days=28):
    """Daily MAE per model, and its rolling mean over the last window_days calendar days.

    RANGE over julianday counts days, not rows: with ROWS, a missing day would silently
    stretch the window. Daily MAEs come from each model's own SPs, so this is a
    monitoring view; the contest scores use the common sample.
    """
    sql = f"""
    WITH daily AS (
        SELECT f.model, f.settlement_date AS day,
               AVG(ABS(o.demand_mw - f.forecast_mw)) AS mae,
               COUNT(*) AS n
        FROM forecasts AS f
        JOIN outturn AS o
          ON o.settlement_date = f.settlement_date AND o.settlement_period = f.settlement_period
        GROUP BY f.model, f.settlement_date
    )
    SELECT model, day, mae, n,
           AVG(mae) OVER (
               PARTITION BY model
               ORDER BY julianday(day)
               RANGE BETWEEN {int(window_days) - 1} PRECEDING AND CURRENT ROW
           ) AS rolling_mae
    FROM daily
    ORDER BY model, day
    """
    return pd.read_sql_query(sql, con)
