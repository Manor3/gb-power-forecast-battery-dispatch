"""Battery arbitrage: one-day perfect-foresight LP (a MIP with binary=True).

Constraint numbers match the formulation:
(1) SoC dynamics   (2) energy limits   (3) power limits   (4) terminal SoC
(5) mode binary, MIP only   (6) daily cycle cap, optional 
(7) no price, no trade: c_t = d_t = 0 for t ∈ Z, the periods with a NaN price
Objective: max sum over t ∉ Z of p_t * (d_t - c_t) * dt
"""
import math

import pyomo.environ as pyo


def to_period_dict(values):
    """List of per-period values -> dict keyed 1..N, the shape build_model expects."""
    return {t: v for t, v in enumerate(values, start=1)}


def build_model(prices, dt, P_max, E_max, eta_c, eta_d, e0, binary=False, n_cycles=None):
    m = pyo.ConcreteModel()
    N = len(prices)
    m.T = pyo.RangeSet(1, N)

    # variables; their bounds are constraints (2) energy and (3) power
    m.c = pyo.Var(m.T, bounds=(0, P_max))
    m.d = pyo.Var(m.T, bounds=(0, P_max))
    m.e = pyo.Var(m.T, bounds=(0, E_max))

    # (1) SoC dynamics, one per period
    def soc_rule(m, t):
        prev = e0 if t == 1 else m.e[t-1]
        return m.e[t] == prev + (eta_c*m.c[t] - m.d[t]/eta_d) * dt

    m.soc = pyo.Constraint(m.T, rule=soc_rule)

    # (4) terminal condition: end the day where it started
    m.terminal = pyo.Constraint(expr= m.e[N]==e0)

    # (5) MIP only: mode binary, so no charging and discharging in the same period
    if binary:
        m.u = pyo.Var(m.T, domain=pyo.Binary)

        def charge_mode_rule(m, t):
            return m.c[t] <= P_max * m.u[t]

        def discharge_mode_rule(m, t):
            return m.d[t] <= P_max * (1 - m.u[t])

        m.charge_mode = pyo.Constraint(m.T, rule=charge_mode_rule)
        m.discharge_mode = pyo.Constraint(m.T, rule=discharge_mode_rule)

    # (6) optional daily cycle cap: cell-side energy discharged <= n_cycles full cycles
    if n_cycles is not None:
        m.cycle_cap = pyo.Constraint(
            expr=sum(m.d[t] / eta_d * dt for t in m.T) <= n_cycles * E_max
        )

        # (7) no price, no trade: Z = periods with a NaN price; idle in all of them
    m.Z = pyo.Set(within=m.T, initialize=[t for t in m.T if math.isnan(prices[t])])

    def no_price_charge_rule(m, t):
        return m.c[t] == 0

    def no_price_discharge_rule(m, t):
        return m.d[t] == 0

    m.no_price_charge = pyo.Constraint(m.Z, rule=no_price_charge_rule)
    m.no_price_discharge = pyo.Constraint(m.Z, rule=no_price_discharge_rule)

    # objective: arbitrage revenue in £, priced periods only (NaN * 0 is still NaN)
    m.obj = pyo.Objective(
        expr=sum(prices[t] * (m.d[t] - m.c[t]) * dt for t in m.T if t not in m.Z),
        sense=pyo.maximize,
    )

    return m


def run(m, verbose=False):
    """Solve with HiGHS, raise if not optimal, and return the objective in £."""
    res = pyo.SolverFactory("appsi_highs").solve(m)
    tc = res.solver.termination_condition
    if tc != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"solver returned {tc}")
    if verbose:
        print(tc)
        print("c:", {t: pyo.value(m.c[t]) for t in m.T})
        print("d:", {t: pyo.value(m.d[t]) for t in m.T})
        print("e:", {t: pyo.value(m.e[t]) for t in m.T})
        print("objective:", pyo.value(m.obj))
    return pyo.value(m.obj)


def cycles_used(m, eta_d, dt, E_max):
    """Full equivalent cycles in the solved day: cell-side energy discharged / E_max."""
    return sum(pyo.value(m.d[t]) / eta_d * dt for t in m.T) / E_max


def check_solution(m, dt, P_max, E_max, eta_c, eta_d, e0, binary=False, n_cycles=None, tol=1e-6):
    """Runtime checks on a solved model (the formulation's invariants).

    These are the dashboard lights: they run on every real solve and raise
    ValueError listing every check that failed.
    """
    c = [pyo.value(m.c[t]) for t in m.T]
    d = [pyo.value(m.d[t]) for t in m.T]
    e = [pyo.value(m.e[t]) for t in m.T]
    failed = []

    if any(x < -tol or x > P_max + tol for x in c + d):
        failed.append("power outside [0, P_max]")
    if any(x < -tol or x > E_max + tol for x in e):
        failed.append("SoC outside [0, E_max]")
    if abs(e[-1] - e0) > tol:
        failed.append(f"end SoC {e[-1]:.6f} differs from e0 = {e0}")

    # the SoC path rebuilt from c and d must match the solver's e, period by period
    prev = e0
    for ci, di, ei in zip(c, d, e):
        if abs(ei - (prev + (eta_c * ci - di / eta_d) * dt)) > tol:
            failed.append("SoC path doesn't follow constraint (1)")
            break
        prev = ei

    # over a day that ends where it started, energy out = eta_c * eta_d * energy in
    if abs(sum(d) - eta_c * eta_d * sum(c)) > tol * len(c):
        failed.append("energy identity broken: sum(d) != eta_c * eta_d * sum(c)")

    obj = pyo.value(m.obj)
    if not math.isfinite(obj):
        failed.append(f"revenue is {obj}: a NaN price reached the objective")
    elif obj < -tol:
        failed.append("negative revenue, but idling earns 0")

    # (7) no trading in a period with no price
    if any(pyo.value(m.c[t]) > tol or pyo.value(m.d[t]) > tol for t in m.Z):
        failed.append("trading in a period with no price, constraint (7)")

    if binary and any(min(ci, di) > tol for ci, di in zip(c, d)):
        failed.append("charging and discharging in the same period")

    if n_cycles is not None and cycles_used(m, eta_d, dt, E_max) > n_cycles + tol:
        failed.append(f"more than {n_cycles} cycles")

    if failed:
        raise ValueError("solution checks failed: " + "; ".join(failed))


if __name__ == "__main__":
    # Runs only when you run this file directly, never when the tests import it.
    m = build_model(prices={1: 40, 2: 60}, dt=0.5, P_max=1, E_max=2,
                    eta_c=1, eta_d=0.8, e0=0)
    run(m, verbose=True)
