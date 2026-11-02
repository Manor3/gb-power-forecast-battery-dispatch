"""Point-in-time rules on toy lists of publish times."""
import pandas as pd
import pytest

from pit import assert_no_leak, pick_vintage, trainable

DAY = pd.Timestamp("2025-01-15")
TAU = pd.Timestamp("2025-01-14 09:30", tz="UTC")  # 09:30 GMT on D-1
TAUS = pd.Series({DAY: TAU})


def vintage(published, sp, mw, day=DAY):
    return dict(settlement_date=day, settlement_period=sp, publish_time=pd.Timestamp(published, tz="UTC"), demand_mw=mw)


def test_last_vintage_published_by_tau_wins():
    ndf = pd.DataFrame([
        vintage("2025-01-14 09:05", 1, 100.0),
        vintage("2025-01-14 09:30", 1, 200.0),   # exactly at τ: published by τ, so allowed
        vintage("2025-01-14 12:05", 1, 300.0),   # after τ: never
    ])
    got = pick_vintage(ndf, TAUS)
    assert got["demand_mw"].tolist() == [200.0]


def test_vintage_rule_is_applied_per_settlement_period():
    ndf = pd.DataFrame([
        vintage("2025-01-14 09:05", 1, 100.0),
        vintage("2025-01-14 09:05", 2, 110.0),
        vintage("2025-01-14 09:20", 1, 150.0),   # a partial update that only covers SP 1
    ])
    got = pick_vintage(ndf, TAUS).set_index("settlement_period")["demand_mw"]
    assert got.to_dict() == {1: 150.0, 2: 110.0}


def test_a_day_with_only_late_vintages_gets_no_neso_entry():
    ndf = pd.DataFrame([vintage("2025-01-14 12:05", 1, 300.0)])
    assert pick_vintage(ndf, TAUS).empty


def test_trainable_needs_the_target_and_every_input_published():
    tau = pd.Timestamp("2025-01-14 09:30", tz="UTC")
    target = pd.Series(pd.to_datetime(["2025-01-14 09:00", "2025-01-14 09:00", "2025-01-14 10:00"], utc=True))
    inputs = pd.Series(pd.to_datetime(["2025-01-13 00:00", "2025-01-19 00:00", "2025-01-13 00:00"], utc=True))
    # row 0: fine; row 1: its reanalysis weather lands 5 days later; row 2: target not yet published
    assert trainable(target, inputs, tau).tolist() == [True, False, False]


def test_leak_check():
    ok = pd.Series(pd.to_datetime(["2025-01-14 09:00", None], utc=True))  # NaT: no input used, not a leak
    assert_no_leak(ok, TAU, "fine")
    late = pd.Series(pd.to_datetime(["2025-01-14 09:00", "2025-01-14 09:31"], utc=True))
    with pytest.raises(ValueError, match="1 rows"):
        assert_no_leak(late, TAU, "late")
