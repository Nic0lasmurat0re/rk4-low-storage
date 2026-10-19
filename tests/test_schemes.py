import tracemalloc

import numpy as np
import pytest

from lsrk import solve
from lsrk.problems import coupled_rhs, initial_state
from lsrk.schemes import METHODS
from lsrk.stability import real_axis_limit, stability_polynomial

ORDERS = {"euler": 1, "rk2": 2, "rk4": 4, "rk4_ls": 4, "rk4_ls_numexpr": 4}


def linear_rhs(t, y):
    return -y + np.sin(t)


def linear_exact(t):
    return 0.5 * (np.sin(t) - np.cos(t)) + 1.5 * np.exp(-t)


@pytest.mark.parametrize("method", METHODS)
def test_observed_order(method):
    errors = [abs(solve(linear_rhs, (0, 5), [1.0], n, method)[1][-1, 0] - linear_exact(5)) for n in (40, 80, 160)]
    order = np.log2(errors[1] / errors[2])
    assert order == pytest.approx(ORDERS[method], abs=0.15)


@pytest.mark.parametrize("method", ["rk4", "rk4_ls"])
def test_cubic_quadrature_is_exact(method):
    # y' = f(t) only: an order-4 scheme integrates cubics exactly, which checks the stage times c_s
    _, y = solve(lambda t, y: np.full_like(y, 4 * t**3 - 3 * t**2 + 1), (0, 2), [0.0], 3, method)
    assert y[-1, 0] == pytest.approx(16 - 8 + 2, abs=1e-12)


def test_low_storage_stability_polynomial_matches_taylor_to_order_4():
    coef = stability_polynomial("rk4_ls").coef
    assert coef[:5] == pytest.approx([1, 1, 1 / 2, 1 / 6, 1 / 24], abs=1e-12)


def test_real_axis_limits():
    assert real_axis_limit("euler") == pytest.approx(2.0)
    assert real_axis_limit("rk4") == pytest.approx(2.785, abs=1e-3)
    assert real_axis_limit("rk4_ls") == pytest.approx(4.657, abs=1e-3)


def test_numexpr_variant_matches_numpy():
    y0 = initial_state(1000)
    _, a = solve(coupled_rhs, (0, 1), y0, 50, "rk4_ls")
    _, b = solve(coupled_rhs, (0, 1), y0, 50, "rk4_ls_numexpr")
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_save_every_keeps_initial_and_final_states():
    t, y = solve(linear_rhs, (0, 1), [1.0], 10, "rk4", save_every=3)
    assert t.tolist() == pytest.approx([0, 0.3, 0.6, 0.9, 1.0])
    assert y.shape == (5, 1)


def peak_vectors(method, n=200_000):
    y = initial_state(n)
    tracemalloc.start()
    step = METHODS[method](n)
    for i in range(3):
        step(coupled_rhs, 0.01 * i, y, 0.01)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak / (8 * n)


def test_work_arrays():
    f_cost = peak_vectors("euler")  # Euler keeps no work array
    assert peak_vectors("rk4") - f_cost == pytest.approx(5, abs=0.1)
    assert peak_vectors("rk4_ls") - f_cost == pytest.approx(1, abs=0.1)
