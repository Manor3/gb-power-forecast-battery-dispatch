"""End to end on synthetic data: every model forecasts every SP, NESO is graded on the
vintage it had at τ, and the GBM beats the naive when weather drives demand."""
from datetime import date

import pandas as pd
import pytest

import backtest
import evaluate
import gbm
import synthetic

FAST = dict(gbm.PARAMS, n_estimators=150, learning_rate=0.1, min_child_samples=50)
TEST_DAYS = pd.date_range("2024-01-01", "2024-02-29").date


@pytest.fixture(scope="module")
def run():
    d = synthetic.make(date(2023, 9, 1), date(2024, 2, 29))
    forecasts, panel, log = backtest.run_backtest(d.outturn, d.wx_act, d.wx_fc, d.ndf, TEST_DAYS,
                                                  date(2023, 9, 15), "v1", params=FAST)
    wide = forecasts.pivot_table(index=["settlement_date", "settlement_period"], columns="model",
                                 values="forecast_mw").reset_index()
    wide = wide.merge(d.outturn[["settlement_date", "settlement_period", "demand_mw"]])
    wide = wide.rename(columns={"demand_mw": "actual"})
    wide.columns.name = None
    return forecasts, panel, log, wide


def test_every_model_forecasts_every_sp(run):
    forecasts, panel, _, _ = run
    n_sps = int(panel["settlement_date"].isin(pd.to_datetime(TEST_DAYS)).sum())
    counts = forecasts["model"].value_counts().to_dict()
    assert counts == {m: n_sps for m in ["naive", "neso", "gbm_v1", "gbm_v1_actualwx"]}


def test_neso_is_graded_on_its_0905_vintage_not_the_better_1205_one(run):
    forecasts, _, _, _ = run
    neso = forecasts[forecasts["model"] == "neso"]
    assert (neso["source_time"] <= neso["issue_time"]).all()
    assert set(neso["source_time"].dt.tz_convert("Europe/London").dt.strftime("%H:%M")) == {"09:05"}


def test_refits_never_learn_from_unpublished_rows(run):
    _, _, log, _ = run
    refits = [entry["message"] for entry in log if entry["stage"] == "refit"]
    assert len(refits) == 2  # January and February
    assert "newest 2023-12-26" in refits[0]  # τ = 31 Dec 09:30; reanalysis lands 5 days late


def test_ranking_on_synthetic_data(run):
    _, _, _, wide = run
    s = evaluate.summary(wide, ["naive", "neso", "gbm_v1", "gbm_v1_actualwx"], scale=1000.0)
    assert s.loc["gbm_v1", "MAE"] < s.loc["naive", "MAE"]          # weather is wired in correctly
    assert s.loc["gbm_v1_actualwx", "MAE"] < s.loc["gbm_v1", "MAE"]  # real weather beats forecasts
    assert s.loc["neso", "MAE"] == pytest.approx(600 * 0.798, rel=0.1)  # the 09:05 noise, not 12:05's
