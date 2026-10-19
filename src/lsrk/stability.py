"""Linear stability of the schemes: stability polynomial R(z) and real-axis limit.

Applying one step of a scheme to y' = lambda y with h = 1 gives y1 = R(z) y0 with
z = h * lambda. The step is stable when |R(z)| <= 1.
"""

import numpy as np
from numpy.polynomial import Polynomial

from .schemes import CK_A, CK_B


def stability_polynomial(method):
    """R(z) as a numpy Polynomial, computed by running the scheme on symbolic z."""
    one = Polynomial([1.0])
    z = Polynomial([0.0, 1.0])
    if method == "euler":
        return one + z
    if method == "rk2":
        return one + z + z**2 / 2
    if method == "rk4":
        return one + z + z**2 / 2 + z**3 / 6 + z**4 / 24
    if method in ("rk4_ls", "rk4_ls_numexpr"):
        y, d = one, Polynomial([0.0])
        for a, b in zip(CK_A, CK_B):
            d = a * d + z * y
            y = y + b * d
        return y
    raise ValueError(f"unknown method {method!r}")


def real_axis_limit(method):
    """Largest r such that |R(x)| <= 1 for every x in [-r, 0]."""
    R = stability_polynomial(method)
    # boundary points are real roots of R(x) = 1 or R(x) = -1 on the negative axis
    candidates = np.concatenate([(R - 1).roots(), (R + 1).roots()])
    candidates = sorted(-c.real for c in candidates if abs(c.imag) < 1e-9 and c.real < -1e-9)
    for r in candidates:
        # the stable interval ends at the first boundary point past which |R| exceeds 1
        if abs(R(-r - 1e-6)) > 1:
            return r
    return np.inf
