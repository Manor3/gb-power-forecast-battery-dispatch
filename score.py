"""Forecast scores. Error convention everywhere: error = actual - forecast.

Every function refuses bad input (NaN, mismatched lengths, empty) rather than return a
number: pandas' .mean() skips NaN silently, so a missing half-hour would vanish unseen.
"""
import math

import numpy as np


def _diff(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"length mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("nothing to score")
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("missing values: align and filter upstream, don't score around gaps")
    return a - b


def mae(actual, forecast):
    """Mean absolute error: the typical miss, in MW."""
    return float(np.mean(np.abs(_diff(actual, forecast))))


def rmse(actual, forecast):
    """Root mean squared error: the typical miss with big misses counted extra, in MW."""
    return float(np.sqrt(np.mean(_diff(actual, forecast) ** 2)))


def bias(actual, forecast):
    """Mean error. Positive: the forecast is too low on average."""
    return float(np.mean(_diff(actual, forecast)))


def mase(actual, forecast, scale):
    """MAE divided by a scale fixed in advance: the naive's MAE over the training years."""
    if not scale > 0:
        raise ValueError(f"MASE scale must be positive, got {scale}")
    return mae(actual, forecast) / scale


def skill(actual, forecast, naive):
    """Share of the naive's error removed on the same half-hours: 1 - MAE / MAE_naive."""
    return 1 - mae(actual, forecast) / mae(actual, naive)


def dm_test(loss_a, loss_b, max_lag=None):
    """Diebold-Mariano test on per-day losses of forecasters A and B (same days, same order).

    d = loss_a - loss_b, so a negative mean means A had the lower loss. Days are serially
    correlated (weather regimes last), so the variance of the mean uses Newey-West with
    Bartlett weights rather than assuming independent days.

    Returns (mean difference, DM statistic, two-sided p-value from the normal).
    """
    d = _diff(loss_a, loss_b)
    n = d.size
    if n < 2:
        raise ValueError("need at least 2 days")
    if max_lag is None:
        max_lag = int(4 * (n / 100) ** (2 / 9))  # Newey-West rule of thumb: 5 for a year
    mean = float(d.mean())
    dev = d - mean
    long_run_var = dev @ dev / n
    for k in range(1, min(max_lag, n - 1) + 1):
        long_run_var += 2 * (1 - k / (max_lag + 1)) * (dev[k:] @ dev[:-k]) / n
    if long_run_var <= 0:  # every day had the same difference
        return mean, (0.0 if mean == 0 else math.copysign(math.inf, mean)), (1.0 if mean == 0 else 0.0)
    stat = mean / math.sqrt(long_run_var / n)
    return mean, float(stat), float(math.erfc(abs(stat) / math.sqrt(2)))
