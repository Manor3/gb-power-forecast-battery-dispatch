"""Hand-checked test cases for score.py. Run with: python -m pytest -v

The toy case is the one you solved by hand: four half-hours, numbers in GW.
Error convention everywhere: error = actual - forecast (positive = forecast too low).
"""
import math

import numpy as np
import pytest

from score import bias, dm_test, mae, mase, rmse, skill

TOL = 1e-9

ACTUAL = [30, 32, 34, 36]
NAIVE = [26, 36, 30, 40]
MODEL_A = [32, 30, 36, 34]
MODEL_B = [30, 32, 34, 44]
TRAIN_SCALE = 5.0  # the naive's MAE over the training years, fixed in advance

ALL_METRICS = ["mae", "rmse", "bias", "mase", "skill"]


def call(name, actual, forecast):
    """Call one metric by name, passing whatever extra argument it needs."""
    if name == "mase":
        return mase(actual, forecast, TRAIN_SCALE)
    if name == "skill":
        return skill(actual, forecast, NAIVE)
    return {"mae": mae, "rmse": rmse, "bias": bias}[name](actual, forecast)


# --- the values you solved by hand ---

@pytest.mark.parametrize(
    "forecast, exp_mae, exp_rmse, exp_bias, exp_skill, exp_mase",
    [
        pytest.param(NAIVE, 4.0, 4.0, 0.0, 0.0, 0.8, id="naive"),
        pytest.param(MODEL_A, 2.0, 2.0, 0.0, 0.5, 0.4, id="A-steady"),
        pytest.param(MODEL_B, 2.0, 4.0, -2.0, 0.5, 0.4, id="B-one-big-miss"),
    ],
)
def test_toy_case(forecast, exp_mae, exp_rmse, exp_bias, exp_skill, exp_mase):
    assert mae(ACTUAL, forecast) == pytest.approx(exp_mae, abs=TOL)
    assert rmse(ACTUAL, forecast) == pytest.approx(exp_rmse, abs=TOL)
    assert bias(ACTUAL, forecast) == pytest.approx(exp_bias, abs=TOL)
    assert skill(ACTUAL, forecast, NAIVE) == pytest.approx(exp_skill, abs=TOL)
    assert mase(ACTUAL, forecast, TRAIN_SCALE) == pytest.approx(exp_mase, abs=TOL)


def test_perfect_forecast():
    """No error anywhere: every metric is 0, and skill is 1 (all the naive's error removed)."""
    for name in ["mae", "rmse", "bias", "mase"]:
        assert call(name, ACTUAL, ACTUAL) == pytest.approx(0.0, abs=TOL)
    assert skill(ACTUAL, ACTUAL, NAIVE) == pytest.approx(1.0, abs=TOL)


def test_bias_sign_convention():
    """Forecast too low gives a positive bias; too high gives a negative one."""
    assert bias([30, 30], [28, 28]) == pytest.approx(2.0, abs=TOL)
    assert bias([30, 30], [33, 33]) == pytest.approx(-3.0, abs=TOL)


def test_accepts_numpy_arrays():
    """Same answer whether the data arrives as lists or numpy arrays."""
    assert mae(np.array(ACTUAL), np.array(MODEL_B)) == pytest.approx(2.0, abs=TOL)


# --- invariants: true for any data, like check_solution in Session 1 ---

def test_ordering_holds_on_random_data():
    """Always: |bias| <= MAE <= RMSE <= sqrt(n) * MAE.

    MAE = RMSE only when every miss is the same size (the naive and A).
    RMSE = sqrt(n) * MAE only when one miss carries all the error (B, with n = 4).
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        actual = rng.normal(30, 5, size=48)
        forecast = actual + rng.standard_t(df=3, size=48)  # fat-tailed errors
        b, m, r = bias(actual, forecast), mae(actual, forecast), rmse(actual, forecast)
        assert abs(b) <= m + TOL
        assert m <= r + TOL
        assert r <= math.sqrt(len(actual)) * m + TOL


# --- fail loudly: a bad input must stop the run, never give a number ---

@pytest.mark.parametrize("name", ALL_METRICS)
@pytest.mark.parametrize("where", ["actual", "forecast"])
def test_missing_value_raises(name, where):
    """pandas' .mean() skips NaN silently, so a missing half-hour would vanish from the score."""
    with_gap = [30, float("nan"), 34, 36]
    actual, forecast = (with_gap, MODEL_A) if where == "actual" else (ACTUAL, with_gap)
    with pytest.raises(ValueError):
        call(name, actual, forecast)


@pytest.mark.parametrize("name", ALL_METRICS)
def test_length_mismatch_raises(name):
    """Four actuals and three forecasts means the alignment step went wrong upstream."""
    with pytest.raises(ValueError):
        call(name, ACTUAL, MODEL_A[:3])


@pytest.mark.parametrize("name", ["mae", "rmse", "bias", "mase"])
def test_empty_raises(name):
    """Nothing left to score means every half-hour was excluded: stop and look."""
    with pytest.raises(ValueError):
        call(name, [], [])


# --- Diebold-Mariano, hand-checked ---
# Daily loss differentials d = [1, -1, 1, -1, 2]: mean 0.4, deviations [0.6, -1.4, 0.6, -1.4, 1.6].
# Variance: 7.2 / 5 = 1.44. Lag-1 autocovariance: -4.76 / 5 = -0.952, Bartlett weight 1/2.
D = np.array([1.0, -1.0, 1.0, -1.0, 2.0])


@pytest.mark.parametrize(
    "max_lag, long_run_var",
    [
        pytest.param(0, 1.44, id="no-HAC: days treated as independent"),
        pytest.param(1, 1.44 + 2 * 0.5 * -0.952, id="Newey-West, 1 lag"),
    ],
)
def test_dm_hand_checked(max_lag, long_run_var):
    mean, stat, p = dm_test(D + 10, np.full(5, 10.0), max_lag=max_lag)
    assert mean == pytest.approx(0.4, abs=TOL)
    assert stat == pytest.approx(0.4 / math.sqrt(long_run_var / 5), abs=1e-9)
    assert p == pytest.approx(math.erfc(abs(stat) / math.sqrt(2)), abs=1e-12)


def test_dm_sign_favours_the_lower_loss():
    rng = np.random.default_rng(1)
    loss_b = rng.uniform(100, 200, 365)
    mean, stat, p = dm_test(loss_b - 20 + rng.normal(0, 5, 365), loss_b)
    assert mean < 0 and stat < 0 and p < 0.001


def test_dm_identical_losses():
    assert dm_test([5.0, 6.0, 7.0], [5.0, 6.0, 7.0]) == (0.0, 0.0, 1.0)


@pytest.mark.parametrize("bad", ["nan", "length"])
def test_dm_refuses_bad_input(bad):
    a, b = [1.0, 2.0, 3.0], [1.0, 2.0, float("nan")] if bad == "nan" else [1.0, 2.0]
    with pytest.raises(ValueError):
        dm_test(a, b)
