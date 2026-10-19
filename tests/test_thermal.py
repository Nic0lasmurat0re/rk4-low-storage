import numpy as np
import pytest
from scipy.integrate import solve_ivp

from lsrk import solve
from lsrk.stability import real_axis_limit
from lsrk.thermal import random_model, simulate


@pytest.fixture(scope="module")
def model():
    return random_model(10)


@pytest.fixture(scope="module")
def reference(model):
    T = np.full(model.n, 250.0)
    for start, end, sunlit in model.phases(2):
        T = solve_ivp(model.phase_rhs(sunlit), (start, end), T, method="DOP853", rtol=1e-13, atol=1e-10).y[:, -1]
    return T


@pytest.mark.parametrize("method", ["rk4", "rk4_ls"])
def test_phase_by_phase_integration_is_fourth_order(model, reference, method):
    errors = [np.max(np.abs(simulate(model, np.full(model.n, 250.0), 2, spo, method)[1][-1] - reference))
              for spo in (50, 100)]
    assert np.log2(errors[0] / errors[1]) == pytest.approx(4, abs=0.3)


def test_low_storage_allows_larger_stable_step(model):
    ratio = model.stable_step(real_axis_limit("rk4_ls")) / model.stable_step(real_axis_limit("rk4"))
    assert ratio == pytest.approx(4.657 / 2.785, rel=1e-3)


def test_temperatures_settle_into_a_bounded_orbit(model):
    t, T = simulate(model, np.full(model.n, 250.0), 8, 100, save_every=1)
    last_orbit = T[t >= 7 * model.orbit_period]
    previous_orbit = T[(t >= 6 * model.orbit_period) & (t <= 7 * model.orbit_period)]
    assert np.all((last_orbit > 150) & (last_orbit < 450))  # K
    # cooling in eclipse: radiation to deep space is modelled
    assert last_orbit.min() < last_orbit.max() - 10
    # periodic regime: two consecutive orbits nearly coincide
    assert np.max(np.abs(last_orbit - previous_orbit)) < 2.0


def test_steps_must_split_evenly_between_phases(model):
    with pytest.raises(ValueError):
        simulate(model, np.full(model.n, 250.0), 1, 33)


def test_stepping_across_the_eclipse_jump_drops_rk4_to_first_order(model, reference):
    # Without restarting at each transition, the last RK4 stage (t + h) lands on the jump
    errors = [np.max(np.abs(solve(model.rhs, (0, 2 * model.orbit_period), np.full(model.n, 250.0), 2 * spo, "rk4")[1][-1]
                            - reference)) for spo in (50, 100)]
    assert np.log2(errors[0] / errors[1]) == pytest.approx(1, abs=0.3)
