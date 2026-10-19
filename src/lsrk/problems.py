"""Test problem used in the benchmarks (the coupled system of the original study)."""

import numpy as np


def coupled_rhs(t, y):
    """f(t, y) = 0.5 sin(roll(y)) - 0.7 cos(y) + 0.1 sin(t / 20).

    Every component is coupled to its neighbour. Written to allocate only the
    returned vector and one scratch vector, so that the memory measured in the
    benchmark is dominated by the integrator, not by f.
    """
    out = np.cos(y)
    out *= -0.7
    s = np.roll(y, 1)
    np.sin(s, out=s)
    s *= 0.5
    out += s
    out += 0.1 * np.sin(t / 20)
    return out


def initial_state(n):
    return np.linspace(0.5, 1.5, n)
