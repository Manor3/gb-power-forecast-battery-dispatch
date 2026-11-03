"""The database enforces the protocol: append-only, one name per attempt, 2025 once."""
import shutil
import sqlite3
import subprocess

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


def test_protocol_names_what_changed(con):
    locked = dict(CONFIG, code={"gbm.py": "a1"}, packages={"lightgbm": "4.7.0"})
    store.save_run(con, "gbm_v1", 2024, locked, forecasts("gbm_v1", [1.0]))
    with pytest.raises(ValueError, match=r"in code: gbm\.py\. .*frozen"):
        store.check_protocol(con, "gbm_v1", 2025, dict(locked, code={"gbm.py": "b2"}), 2024, 2025)  # edited code
    with pytest.raises(ValueError, match="in packages: lightgbm"):
        store.check_protocol(con, "gbm_v1", 2025, dict(locked, packages={"lightgbm": "4.8.0"}), 2024, 2025)  # new library
    store.check_protocol(con, "gbm_v1", 2025, locked, 2024, 2025)  # same code, same libraries: allowed


def test_blob_id_is_gits_on_any_os(tmp_path):
    """A Windows checkout writes CRLF, Linux writes LF: both give the ID git stores for "hello\\n"."""
    (tmp_path / "lf.py").write_bytes(b"hello\n")
    (tmp_path / "crlf.py").write_bytes(b"hello\r\n")
    assert store.blob_id(tmp_path / "lf.py") == store.blob_id(tmp_path / "crlf.py") == "ce013625030ba8dba906f756967f9e9ca394464a"


@pytest.mark.skipif(shutil.which("git") is None or not (store.REPO / ".git").exists(),
                    reason="needs git on PATH and a git checkout: CI has both")
def test_blob_ids_match_git_for_every_module():
    files = sorted(p.name for p in store.REPO.glob("*.py"))
    git = subprocess.run(["git", "hash-object", "--", *files], cwd=store.REPO, capture_output=True, text=True, check=True)
    assert git.stdout.split() == [store.blob_id(store.REPO / f) for f in files]


def test_fingerprint_follows_imports(tmp_path, monkeypatch):
    (tmp_path / "root.py").write_text("import helper\nimport numpy\n")
    (tmp_path / "helper.py").write_text("import json\nimport requests\nfrom pandas import Timestamp\n")
    (tmp_path / "unrelated.py").write_text("import lightgbm\n")
    monkeypatch.setattr(store, "REPO", tmp_path)
    fp = store.fingerprint("root")
    assert set(fp["code"]) == {"root.py", "helper.py"}  # imports are followed; unrelated.py is never reached
    assert set(fp["packages"]) == {"numpy", "pandas"}   # json is stdlib, requests is UNLOCKED
    before = fp["code"]["helper.py"]
    (tmp_path / "helper.py").write_text("import json\nimport requests\nfrom pandas import Timestamp\nX = 1\n")
    assert store.fingerprint("root")["code"]["helper.py"] != before  # an edit in an imported module shows
