"""The database enforces the protocol: append-only, one name per attempt, 2025 once."""
import sqlite3

import pandas as pd
import pytest

import store

CONFIG = dict(version="v1", features=["tod", "naive"])


@pytest.fixture
def con():
    c = store.connect(":memory:")
    yield c
    c.close()


def forecasts(model, values, day="2024-01-15"):
    return pd.DataFrame({
        "model": model,
        "issue_time": pd.Timestamp("2024-01-14 09:30", tz="UTC"),
        "settlement_date": pd.Timestamp(day),
        "settlement_period": range(1, len(values) + 1),
        "forecast_mw": values,
        "source_time": pd.NaT,
    })


def test_a_forecast_once_written_cannot_be_rewritten(con):
    store.save_run(con, "gbm_v1", 2024, CONFIG, forecasts("gbm_v1", [100.0, 110.0]))
    with pytest.raises(sqlite3.IntegrityError):
        store.save_run(con, "gbm_v1b", 2024, CONFIG, forecasts("gbm_v1", [999.0, 999.0]))
    # the failed run left nothing behind: no run row, no forecasts
    assert con.execute("SELECT COUNT(*) FROM runs WHERE model = 'gbm_v1b'").fetchone()[0] == 0
    assert con.execute("SELECT MAX(forecast_mw) FROM forecasts").fetchone()[0] == 110.0


def test_benchmarks_are_written_once_and_then_skipped(con):
    store.save_run(con, "gbm_v1", 2024, CONFIG, pd.concat([forecasts("naive", [1.0]), forecasts("gbm_v1", [2.0])]))
    store.save_run(con, "gbm_v2", 2024, CONFIG, pd.concat([forecasts("naive", [1.0]), forecasts("gbm_v2", [3.0])]))
    assert con.execute("SELECT COUNT(*) FROM forecasts WHERE model = 'naive'").fetchone()[0] == 1


def test_protocol(con):
    with pytest.raises(ValueError, match="frozen"):
        store.check_protocol(con, "gbm_v1", 2025, CONFIG, 2024, 2025)      # never run on 2024
    store.save_run(con, "gbm_v1", 2024, CONFIG, forecasts("gbm_v1", [1.0]))
    assert store.existing_run(con, "gbm_v1", 2024) == store.config_json(CONFIG)
    assert store.existing_run(con, "gbm_v1", 2025) is None
    with pytest.raises(ValueError, match="never overwritten"):
        store.check_protocol(con, "gbm_v1", 2024, CONFIG, 2024, 2025)      # same name twice
    with pytest.raises(ValueError, match="frozen"):
        store.check_protocol(con, "gbm_v1", 2025, dict(CONFIG, features=["tod"]), 2024, 2025)  # edited settings
    store.check_protocol(con, "gbm_v1", 2025, CONFIG, 2024, 2025)          # frozen settings: allowed
    store.save_run(con, "gbm_v1", 2025, CONFIG, forecasts("gbm_v1", [1.0], day="2025-01-15"))
    store.save_run(con, "gbm_v2", 2024, CONFIG, forecasts("gbm_v2", [1.0]))
    with pytest.raises(ValueError, match="once"):
        store.check_protocol(con, "gbm_v2", 2025, CONFIG, 2024, 2025)      # second look at 2025


def test_rolling_mae_counts_days_not_rows(con):
    """Days 1, 2 and 4 (day 3 missing), 2-day window. RANGE: day 4 averages day 4 only.
    ROWS would average days 2 and 4, a 3-day window labelled as 2."""
    outturn = pd.DataFrame({"settlement_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"]),
                            "settlement_period": 1, "demand_mw": 100.0, "publish_time": pd.NaT})
    store.upsert_outturn(con, outturn)
    rows = pd.concat([forecasts("m", [100.0 - mae], day) for day, mae in
                      [("2024-01-01", 10.0), ("2024-01-02", 20.0), ("2024-01-04", 40.0)]])
    store.save_run(con, "m", 2024, CONFIG, rows)
    got = store.rolling_mae(con, window_days=2).set_index("day")["rolling_mae"]
    assert got.to_dict() == {"2024-01-01": 10.0, "2024-01-02": 15.0, "2024-01-04": 40.0}


def test_read_scored_lines_up_models_and_outturn(con):
    outturn = pd.DataFrame({"settlement_date": pd.Timestamp("2024-01-15"), "settlement_period": [1, 2],
                            "demand_mw": [100.0, 200.0], "publish_time": pd.NaT})
    store.upsert_outturn(con, outturn)
    store.save_run(con, "gbm_v1", 2024, CONFIG, pd.concat([forecasts("naive", [90.0, 190.0]), forecasts("gbm_v1", [95.0])]))
    t = store.read_scored(con, ["naive", "gbm_v1"], pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-12-31").date())
    assert t["actual"].tolist() == [100.0, 200.0]
    assert t["naive"].tolist() == [90.0, 190.0]
    assert t["gbm_v1"].iloc[0] == 95.0 and pd.isna(t["gbm_v1"].iloc[1])
