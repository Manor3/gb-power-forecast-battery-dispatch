"""Point-in-time rules.

The rule that does all the work: an input is usable at issue time τ only if it was
PUBLISHED by τ. Published, not observed. Every row is a pair (issue time, target SP),
never a target alone.
"""


def pick_vintage(ndf, taus):
    """NESO's entry for each target day: per SP, the last NDF value published at or before τ.

    ndf:  [settlement_date, settlement_period, publish_time, demand_mw], one row per vintage
    taus: Series settlement_date -> τ
    """
    x = ndf.merge(taus.rename("tau"), left_on="settlement_date", right_index=True, how="inner")
    x = x[x["publish_time"] <= x["tau"]]
    last = x.sort_values("publish_time").groupby(["settlement_date", "settlement_period"]).tail(1)
    return last.drop(columns="tau").sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)


def trainable(target_publish, info_time, tau):
    """Rows a model fitted at τ may learn from: the target AND every input were published by τ."""
    return (target_publish <= tau) & (info_time <= tau)


def assert_no_leak(info_time, tau, what):
    """Raise if any input to a forecast issued at τ was published after τ.

    NaT means the row used no published input (its value is missing), which is not a leak.
    """
    late = info_time > tau
    if late.any():
        worst = info_time[late].max()
        raise ValueError(f"{what}: {int(late.sum())} rows use inputs published after τ = {tau} (latest {worst})")
