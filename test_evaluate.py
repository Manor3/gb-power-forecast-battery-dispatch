"""Scoring tables: the common sample, the MASE scale, the breakdowns and the DM verdicts."""
import numpy as np
import pandas as pd
import pytest

import evaluate

TOL = 1e-9
MODELS = ["naive", "neso", "gbm"]


def table(day, sps, actual, naive, neso, gbm):
    return pd.DataFrame({"settlement_date": pd.Timestamp(day), "settlement_period": sps,
                         "actual": actual, "naive": naive, "neso": neso, "gbm": gbm})


def test_common_sample_keeps_only_sps_every_model_has_and_logs_the_rest():
    nan = float("nan")
    t = table("2024-01-15", [1, 2, 3, 4], [10, 20, 30, 40], [11, 21, 31, 41], [12, nan, 32, 42], [13, 23, nan, nan])
    t.loc[3, "actual"] = nan  # SP 4 has no outturn either
    log = []
    kept = evaluate.common_sample(t, MODELS, log)
    assert kept["settlement_period"].tolist() == [1]
    messages = " | ".join(entry["message"] for entry in log)
    assert "1 SPs on 1 days dropped: no actual" in messages
    assert "1 SPs on 1 days dropped: no neso" in messages
    assert "2 SPs on 1 days dropped: no gbm" in messages


def test_summary_reproduces_the_hand_solved_toy_case():
    """Step 1's four half-hours, now through the table: A is the 'gbm' column, B the 'neso' one."""
    t = table("2024-01-15", [1, 2, 3, 4], [30, 32, 34, 36], [26, 36, 30, 40], [30, 32, 34, 44], [32, 30, 36, 34])
    s = evaluate.summary(t, MODELS, scale=5.0)
    assert s.loc["gbm", ["MAE", "RMSE", "bias", "MASE", "skill"]].tolist() == pytest.approx([2, 2, 0, 0.4, 0.5])
    assert s.loc["neso", ["MAE", "RMSE", "bias", "RMSE/MAE"]].tolist() == pytest.approx([2, 4, -2, 2])
    assert s.loc["naive", ["MASE", "skill"]].tolist() == pytest.approx([0.8, 0.0])
    assert s["gap_closed"].tolist() == pytest.approx([0.0, 1.0, 1.0])  # naive 0, NESO 1 by definition


def test_mase_scale_uses_only_the_training_years():
    panel = pd.DataFrame({"settlement_date": pd.to_datetime(["2023-12-31", "2023-12-31", "2024-01-01"]),
                          "demand_mw": [100.0, 100.0, 100.0], "naive": [90.0, 110.0, 5000.0]})
    assert evaluate.mase_scale(panel, "2023-01-01", "2024-01-01") == pytest.approx(10.0)


def test_breakdowns_label_half_hours_by_clock_time():
    """On the 46-SP day SP 3 is 02:00. Grouping by SP number would file it under 01:00."""
    n = 46
    t = table("2025-03-30", range(1, n + 1), np.full(n, 100.0), np.full(n, 90.0), np.full(n, 95.0), np.full(n, 99.0))
    g = evaluate.add_groups(t)
    assert g.loc[g["settlement_period"] == 3, "half_hour"].item() == "02:00"
    assert "01:00" not in set(g["half_hour"])
    assert set(g["day_type"]) == {"weekend"} and set(g["month"]) == {"03 Mar"}
    by_type = evaluate.mae_by(g, MODELS, "day_type")
    assert by_type.loc["weekend"].tolist() == pytest.approx([10.0, 5.0, 1.0, 46])


def test_dm_verdicts():
    rng = np.random.default_rng(0)
    days = pd.date_range("2024-01-01", periods=120)
    frames = []
    for d in days:
        actual = rng.normal(30_000, 2_000, 48)
        naive = actual + rng.normal(0, 1_500, 48)
        frames.append(table(d, range(1, 49), actual, naive, naive, actual + rng.normal(0, 500, 48)))
    t = pd.concat(frames, ignore_index=True)
    dm = evaluate.dm_table(t, [("gbm", "naive"), ("neso", "naive")]).set_index(["pair", "loss"])
    assert dm.loc[("gbm vs naive", "abs"), "verdict"] == "gbm better"
    assert dm.loc[("gbm vs naive", "abs"), "days"] == 120  # days, not 5,760 half-hours
    assert dm.loc[("neso vs naive", "sq"), "verdict"] == "no clear winner"  # identical forecasts
