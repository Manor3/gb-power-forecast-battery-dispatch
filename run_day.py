"""Step 7: solve real GB days as LP and MIP, and check every solution."""
from datetime import date
import math

from battery import build_model, check_solution, cycles_used, run
from elexon import fetch_mid_prices

# 1 MW / 2 MWh battery, 88% round-trip efficiency split evenly, starts and ends half full
PARAMS = dict(dt=0.5, P_max=1.0, E_max=2.0, eta_c=math.sqrt(0.88), eta_d=math.sqrt(0.88), e0=1.0)

DAYS = {
    "ordinary day": date(2025, 6, 18),
    "clocks forward, 46 periods": date(2025, 3, 30),
    "clocks back, 50 periods": date(2025, 10, 26),
}


def solve_day(prices, binary):
    m = build_model(prices=prices, binary=binary, **PARAMS)
    revenue = run(m)
    check_solution(m, binary=binary, **PARAMS)
    return m, revenue


if __name__ == "__main__":
    for label, day in DAYS.items():
        prices = fetch_mid_prices(day)
        _, lp = solve_day(prices, binary=False)
        m, mip = solve_day(prices, binary=True)
        cycles = cycles_used(m, PARAMS["eta_d"], PARAMS["dt"], PARAMS["E_max"])
        print(
            f"{day} ({label}): {len(prices)} periods, "
            f"prices £{min(prices.values()):.2f} to £{max(prices.values()):.2f}, "
            f"LP £{lp:.2f}, MIP £{mip:.2f}, {cycles:.2f} cycles"
        )
