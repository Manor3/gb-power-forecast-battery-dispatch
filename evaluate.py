"""Scoring the contest: the common sample, the tables, and the Diebold-Mariano tests."""
import pandas as pd

import gbtime
import score


def common_sample(table, models, log):
    """Keep half-hours where the outturn AND every model exist, so all are graded on the same SPs.

    Every exclusion is logged; nothing disappears silently.
    """
    need = ["actual", *models]
    for col in need:
        gap = table[col].isna()
        if gap.any():
            days = sorted(table.loc[gap, "settlement_date"].dt.date.unique())
            shown = ", ".join(d.isoformat() for d in days[:8]) + (" ..." if len(days) > 8 else "")
            log.append(dict(date="", stage="score",
                            message=f"{int(gap.sum())} SPs on {len(days)} days dropped: no {col} ({shown})"))
    return table[table[need].notna().all(axis=1)].reset_index(drop=True)


def mase_scale(panel, start, end):
    """The naive's MAE on settlement dates in [start, end): the fixed MASE denominator."""
    dates = panel["settlement_date"]
    rows = (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end)) & panel["demand_mw"].notna() & panel["naive"].notna()
    return score.mae(panel.loc[rows, "demand_mw"], panel.loc[rows, "naive"])


def add_groups(table):
    """Add the breakdown keys: local half-hour, day type and month."""
    grid = gbtime.sp_grid(table["settlement_date"].min(), table["settlement_date"].max())
    out = table.merge(grid[["settlement_date", "settlement_period", "tod"]], on=["settlement_date", "settlement_period"])
    out["half_hour"] = [f"{t // 2:02d}:{30 * (t % 2):02d}" for t in out["tod"]]
    out["day_type"] = gbtime.day_type(out["settlement_date"])
    out["month"] = out["settlement_date"].dt.strftime("%m %b")
    return out


def summary(table, models, scale, naive="naive", benchmark="neso"):
    """Metrics per model. gap_closed: the share of the naive-to-benchmark MAE gap a model removes."""
    rows = {}
    for m in models:
        a, f = table["actual"], table[m]
        rows[m] = {
            "MAE": score.mae(a, f),
            "RMSE": score.rmse(a, f),
            "bias": score.bias(a, f),
            "MASE": score.mase(a, f, scale),
            "skill": score.skill(a, f, table[naive]),
            "RMSE/MAE": score.rmse(a, f) / score.mae(a, f),
        }
        if benchmark in table.columns:
            floor, top = score.mae(a, table[naive]), score.mae(a, table[benchmark])
            rows[m]["gap_closed"] = (floor - score.mae(a, f)) / (floor - top)
    return pd.DataFrame(rows).T


def mae_by(table, models, key):
    """MAE per group for each model (MW), plus the number of half-hours in the group."""
    out = pd.DataFrame({m: (table["actual"] - table[m]).abs().groupby(table[key]).mean() for m in models})
    out["n"] = table.groupby(key).size()
    return out


def daily_loss(table, model, loss):
    err = table["actual"] - table[model]
    per_sp = err.abs() if loss == "abs" else err ** 2
    return per_sp.groupby(table["settlement_date"]).mean()


def dm_table(table, pairs, alpha=0.05):
    """Diebold-Mariano on daily mean losses: about 365 pieces of evidence, not 17,520."""
    rows = []
    for a, b in pairs:
        for loss in ("abs", "sq"):
            la, lb = daily_loss(table, a, loss), daily_loss(table, b, loss)
            mean, stat, p = score.dm_test(la.to_numpy(), lb.to_numpy())
            verdict = (f"{a} better" if mean < 0 else f"{b} better") if p < alpha else "no clear winner"
            rows.append(dict(pair=f"{a} vs {b}", loss=loss, days=len(la), mean_diff=mean, dm_stat=stat, p_value=p, verdict=verdict))
    return pd.DataFrame(rows)
