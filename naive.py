"""The seasonal naive: the same local half-hour, one week earlier. The floor to beat.

The rule in full:
1. Match on local clock time (tod), not SP number. After the clocks go forward,
   SP 3 is 02:00, not 01:00.
2. If that clock time happened twice on D-7 (clocks went back), average the two.
3. If it didn't happen on D-7 (clocks went forward), go back another week, to D-14.

The same lookup builds the lag features ("demand at this clock time 2 days ago").
"""
import numpy as np
import pandas as pd


def shift_local(panel, days, values, times=(), fallback_days=None):
    """For each row of panel: values at the same local clock time `days` earlier.

    panel must hold EVERY settlement period of every date (see gbtime.sp_grid), with
    NaN where data is missing. That separates "this clock time didn't exist that day"
    (rule 3 applies) from "the data is missing" (the result is NaN).

    values: columns averaged over the matching rows; NaN if any match is NaN.
    times:  columns reduced with max, e.g. the latest publish time among the matches.
    Returns a DataFrame aligned to panel.index.
    """
    key = pd.DataFrame({
        "row": np.arange(len(panel)),
        "settlement_date": panel["settlement_date"].to_numpy(),
        "tod": panel["tod"].to_numpy(),
    })
    source = panel[["settlement_date", "tod", *values, *times]]

    def matches(k):
        shifted = key.assign(settlement_date=key["settlement_date"] - pd.Timedelta(days=k))
        return shifted.merge(source, on=["settlement_date", "tod"], how="inner")

    found = matches(days)
    if fallback_days is not None:
        absent = ~key["row"].isin(found["row"])
        backup = matches(fallback_days)
        found = pd.concat([found, backup[backup["row"].isin(key.loc[absent, "row"])]])

    groups = found.groupby("row")
    out = pd.DataFrame(index=pd.RangeIndex(len(panel)))
    for col in values:
        complete = groups[col].count() == groups[col].size()
        out[col] = groups[col].mean().where(complete)
    for col in times:
        out[col] = pd.to_datetime(groups[col].max(), utc=True)  # stays a datetime even if nothing matched
    out.index = panel.index
    return out


def seasonal_naive(panel):
    """Naive forecast for every row of panel ('naive'), and the latest publish time it uses ('naive_info')."""
    out = shift_local(panel, 7, values=["demand_mw"], times=["publish_time"], fallback_days=14)
    return out.rename(columns={"demand_mw": "naive", "publish_time": "naive_info"})
