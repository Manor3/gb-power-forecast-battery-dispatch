"""The whole run script, offline: the three fetchers are swapped for synthetic fakes.

This checks the plumbing the unit tests can't see: caching, the protocol check, the
store, the tables, and what a rerun does.
"""
from datetime import date

import pandas as pd
import pytest

import elexon
import gbm
import run_session2 as rs
import synthetic
import weather

SYN = synthetic.make(date(2023, 8, 1), date(2024, 2, 29))


def fake_indo(start, end, chunk_days=7):
    o = SYN.outturn
    return o[o["settlement_date"].between(pd.Timestamp(start), pd.Timestamp(end))].reset_index(drop=True)


FORECAST_ARCHIVE_START = pd.Timestamp("2024-01-03", tz="UTC")  # like the real ECMWF archive: a late start


def fake_sites(kind, start, end):
    """Every site gets the national value, so the weighted average gives it back."""
    hourly = SYN.real_hourly if kind == "reanalysis" else SYN.forecast_hourly
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    long = hourly[(hourly.index >= lo) & (hourly.index < hi)].rename_axis("time_utc").reset_index()
    long = long.melt(id_vars="time_utc", var_name="variable", value_name="value")
    if kind == "forecast":  # no forecasts on 1-2 January: weather versions lose those days, v0 doesn't
        long.loc[long["time_utc"] < FORECAST_ARCHIVE_START, "value"] = float("nan")
    return pd.concat([long.assign(site=s) for s in weather.SITES], ignore_index=True)


def fake_ndf(day):
    n = SYN.ndf
    return n[n["settlement_date"] == pd.Timestamp(day)].reset_index(drop=True)


def explode(*args, **kwargs):
    raise AssertionError("fetched from the network: the cache should have been used")


@pytest.fixture
def offline(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "DATA", tmp_path / "data")
    monkeypatch.setattr(rs, "DB_PATH", tmp_path / "data" / "session2.db")
    monkeypatch.setattr(rs, "RESULTS", tmp_path / "data" / "results")
    monkeypatch.setattr(rs, "TRAIN_START", date(2023, 9, 15))
    monkeypatch.setattr(rs, "HISTORY_START", date(2023, 8, 15))
    monkeypatch.setattr(rs, "YEAR", 2024)
    monkeypatch.setattr(rs, "year_days", lambda year: list(pd.date_range("2024-01-01", "2024-01-14").date))
    monkeypatch.setattr(elexon, "fetch_indo", fake_indo)
    monkeypatch.setattr(weather, "fetch_sites", fake_sites)
    monkeypatch.setattr(elexon, "fetch_ndf_vintages", fake_ndf)
    for key, value in dict(n_estimators=100, learning_rate=0.1, min_child_samples=50).items():
        monkeypatch.setitem(gbm.PARAMS, key, value)
    return tmp_path / "data", monkeypatch


def test_whole_script_offline(offline, capsys):
    data, monkeypatch = offline
    summary_csv = data / "results" / "2024_gbm_v1_summary.csv"

    rs.main()  # first run: fetches, caches, forecasts, stores, scores
    out = capsys.readouterr().out
    assert "Interview line: in 2024" in out
    assert "NESO is graded on its last forecast published by 09:30 on D-1, most often published at 09:05 (100%)" in out
    assert (data / "indo" / "2023-08.csv").exists() and (data / "ndf" / "2024-01.csv").exists()
    summary = pd.read_csv(summary_csv, index_col=0)
    assert summary.loc["gbm_v1", "MAE"] < summary.loc["naive", "MAE"]
    first_log = (data / "log_2024_gbm_v1.csv").read_text()

    rs.main()  # same version again: re-scores the stored forecasts, never re-forecasts
    out = capsys.readouterr().out
    assert "only re-scores" in out and "settings have changed" not in out
    pd.testing.assert_frame_equal(summary, pd.read_csv(summary_csv, index_col=0))
    assert (data / "log_2024_gbm_v1.csv").read_text() == first_log  # the real run's log survives

    monkeypatch.setitem(gbm.PARAMS, "n_estimators", 99)
    rs.main()  # edited settings under an old name: still only re-scores, and says what changed
    assert "settings have changed since gbm_v1 ran (params: n_estimators)" in capsys.readouterr().out

    monkeypatch.setattr(rs, "MODEL_VERSION", "v9")
    with pytest.raises(ValueError, match="isn't defined"):
        rs.main()  # a name with no feature list behind it

    for module, name in [(elexon, "fetch_indo"), (weather, "fetch_sites"), (elexon, "fetch_ndf_vintages")]:
        monkeypatch.setattr(module, name, explode)
    monkeypatch.setattr(rs, "MODEL_VERSION", "v2")
    rs.main()  # a new attempt runs from the cache alone, and is ranked against v1
    out = capsys.readouterr().out
    assert "Every version run on 2024" in out and "gbm_v2 vs gbm_v1" in out

    monkeypatch.setattr(rs, "MODEL_VERSION", "v0")
    rs.main()  # the no-weather path: it covers 1-2 January, the weather versions don't
    out = capsys.readouterr().out
    assert "gbm_v0 vs gbm_v2" in out
    assert "on the same 576 half-hours" in out  # 12 days x 48: ranked only where every version forecast


def test_the_test_year_refuses_settings_never_run_on_the_validation_year(offline):
    """Relabel the synthetic year as the exam: with no validation run behind it, it must refuse."""
    data, monkeypatch = offline
    monkeypatch.setattr(rs, "DEV_YEAR", 2023)
    monkeypatch.setattr(rs, "TEST_YEAR", 2024)
    with pytest.raises(ValueError, match="frozen"):
        rs.main()
    assert not (data / "results").exists()  # refused before any work


def test_the_lock_covers_the_forecaster_not_the_run_script():
    """This script holds YEAR, which changes between 2024 and 2025 by design: locking it would
    block the very run the protocol allows. The modules that make the forecasts are locked."""
    config = rs.config_for("v1")
    assert "run_session2.py" not in config["code"]
    assert set(config["code"]) == {"backtest.py", "elexon.py", "gbm.py", "gbtime.py", "naive.py", "pit.py", "weather.py"}
    assert set(config["packages"]) == {"holidays", "lightgbm", "numpy", "pandas"}
