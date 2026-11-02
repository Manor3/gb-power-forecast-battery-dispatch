"""Hand-checked test cases for battery.py. Run with: python -m pytest -v"""
import pyomo.environ as pyo
import pytest
import math

from battery import build_model, check_solution, run, to_period_dict

TOL = 1e-6

# Shared test parameters; each case overrides only what differs.
BASE = dict(dt=0.5, P_max=1, E_max=2, eta_c=1, eta_d=0.8, e0=0)

PRICES_A1 = {1: 40, 2: 60}
PRICES_A2 = {1: 40, 2: 45}
PRICES_B = to_period_dict([10] * 5 + [100] * 5)
PRICES_C = {1: -50, 2: -50}
PRICES_D = to_period_dict(([10] * 4 + [100] * 4) * 2)
PRICES_E = {1: math.nan, 2: 20, 3: 50}

def test_e_no_price_no_trade():
    obj, m = solve_case(PRICES_E)
    assert obj == pytest.approx(10.0, abs=TOL)
    assert pyo.value(m.c[1]) == pytest.approx(0, abs=TOL)
    assert pyo.value(m.d[1]) == pytest.approx(0, abs=TOL)

def solve_case(prices, **overrides):
    """Build and solve one case. Keywords in overrides replace the BASE values."""
    m = build_model(prices=prices, **{**BASE, **overrides})
    return run(m), m


@pytest.mark.parametrize(
    "prices, overrides, expected",
    [
        pytest.param(PRICES_A1, dict(eta_d=1), 10.0, id="step1-lossless"),
        pytest.param(PRICES_A1, {}, 4.0, id="A1-spread-beats-losses"),
        pytest.param(PRICES_A2, {}, 0.0, id="A2-spread-too-small"),
        pytest.param(PRICES_B, {}, 140.0, id="B-energy-cap-binds"),
        pytest.param(PRICES_B, dict(binary=True), 140.0, id="B-MIP-equals-LP"),
        pytest.param(PRICES_C, dict(e0=2), 10.0, id="C-LP-artefact"),
        pytest.param(PRICES_C, dict(e0=2, binary=True), 5.0, id="C-MIP-blocks-it"),
        pytest.param(PRICES_D, {}, 280.0, id="D-uncapped"),
        pytest.param(PRICES_D, dict(n_cycles=1), 140.0, id="D-one-cycle"),
        pytest.param(PRICES_D, dict(n_cycles=0), 0.0, id="D-zero-cycles"),
    ],
)
def test_objective(prices, overrides, expected):
    obj, m = solve_case(prices, **overrides)
    assert obj == pytest.approx(expected, abs=TOL)
    check_solution(m, **{**BASE, **overrides})  # the runtime checks must not false-alarm on known-good cases


def test_a1_soc_after_charging():
    """Revenue alone can't tell eta_c from eta_d apart; e[1] can."""
    _, m = solve_case(PRICES_A1)
    assert pyo.value(m.e[1]) == pytest.approx(0.5, abs=TOL)


def test_c_lp_charges_and_discharges_at_once():
    """The LP exploits negative prices by doing both in the same period."""
    _, m = solve_case(PRICES_C, e0=2)
    assert any(min(pyo.value(m.c[t]), pyo.value(m.d[t])) > TOL for t in m.T)


def test_c_mip_never_charges_and_discharges_at_once():
    """The mode binary blocks it."""
    _, m = solve_case(PRICES_C, e0=2, binary=True)
    assert all(min(pyo.value(m.c[t]), pyo.value(m.d[t])) < TOL for t in m.T)
