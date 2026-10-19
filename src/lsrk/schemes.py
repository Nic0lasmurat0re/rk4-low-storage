"""Fixed-step explicit Runge-Kutta integrators.

Every stepper advances the state ``y`` *in place* and owns its work arrays, so the
memory a scheme needs is exactly what it allocates in ``__init__`` plus the vector
returned by each evaluation of ``f``. This is what makes the memory comparison
between classical RK4 and the 2N-storage RK4 meaningful.
"""

import numpy as np

# Carpenter & Kennedy (1994), five-stage fourth-order 2N-storage scheme (solution 3).
# Stage s:  d <- A[s] * d + h * f(t + C[s] * h, y)
#           y <- y + B[s] * d
CK_A = np.array([
    0.0,
    -567301805773.0 / 1357537059087.0,
    -2404267990393.0 / 2016746695238.0,
    -3550918686646.0 / 2091501179385.0,
    -1275806237668.0 / 842570457699.0,
])
CK_B = np.array([
    1432997174477.0 / 9575080441755.0,
    5161836677717.0 / 13612068292357.0,
    1720146321549.0 / 2090206949498.0,
    3134564353537.0 / 4481467310338.0,
    2277821191437.0 / 14882151754819.0,
])
CK_C = np.array([
    0.0,
    1432997174477.0 / 9575080441755.0,
    2526269341429.0 / 6820363962896.0,
    2006345519317.0 / 3224310063776.0,
    2802321613138.0 / 2924317926251.0,
])


class Euler:
    """Explicit Euler, order 1. No work array."""

    stages = 1

    def __init__(self, n):
        pass

    def __call__(self, f, t, y, h):
        k = f(t, y)
        k *= h
        y += k


class Heun:
    """Heun's method (RK2), order 2. Work arrays: k1 and the stage state."""

    stages = 2

    def __init__(self, n):
        self.k1 = np.empty(n)
        self.ys = np.empty(n)

    def __call__(self, f, t, y, h):
        k1, ys = self.k1, self.ys
        k1[:] = f(t, y)
        np.multiply(k1, h, out=ys)
        ys += y
        k1 += f(t + h, ys)
        k1 *= h / 2
        y += k1


class RK4:
    """Classical RK4, order 4. Work arrays: k1..k4 and the stage state (5n floats)."""

    stages = 4

    def __init__(self, n):
        self.k = np.empty((4, n))
        self.ys = np.empty(n)

    def __call__(self, f, t, y, h):
        k1, k2, k3, k4 = self.k
        ys = self.ys
        k1[:] = f(t, y)
        np.multiply(k1, h / 2, out=ys)
        ys += y
        k2[:] = f(t + h / 2, ys)
        np.multiply(k2, h / 2, out=ys)
        ys += y
        k3[:] = f(t + h / 2, ys)
        np.multiply(k3, h, out=ys)
        ys += y
        k4[:] = f(t + h, ys)
        # y += h/6 (k1 + 2 k2 + 2 k3 + k4), reusing k1 as the accumulator
        k2 += k3
        k2 *= 2
        k1 += k2
        k1 += k4
        k1 *= h / 6
        y += k1


class RK4LowStorage:
    """Carpenter-Kennedy 2N-storage RK4, order 4, five stages.

    Only one work array ``d`` besides ``y`` itself. The vector returned by ``f`` is
    reused as scratch for ``B[s] * d`` so that no other temporary is created.
    """

    stages = 5

    def __init__(self, n):
        self.d = np.empty(n)

    def __call__(self, f, t, y, h):
        d = self.d
        for s in range(5):
            k = f(t + CK_C[s] * h, y)
            if s == 0:
                np.multiply(k, h, out=d)
            else:
                d *= CK_A[s]
                k *= h
                d += k
            np.multiply(d, CK_B[s], out=k)
            y += k
            del k  # release it before the next evaluation of f allocates a new one


class RK4LowStorageNumexpr(RK4LowStorage):
    """Same scheme, with the two vector updates evaluated by numexpr on all CPU cores."""

    def __init__(self, n):
        import numexpr

        super().__init__(n)
        self._ne = numexpr
        numexpr.set_num_threads(numexpr.detect_number_of_cores())

    def __call__(self, f, t, y, h):
        ne, d = self._ne, self.d
        for s in range(5):
            k = f(t + CK_C[s] * h, y)
            a, b = CK_A[s], CK_B[s]
            ne.evaluate("a * d + h * k", out=d, casting="same_kind")
            ne.evaluate("y + b * d", out=y, casting="same_kind")
            del k


METHODS = {
    "euler": Euler,
    "rk2": Heun,
    "rk4": RK4,
    "rk4_ls": RK4LowStorage,
    "rk4_ls_numexpr": RK4LowStorageNumexpr,
}


def solve(f, t_span, y0, n_steps, method="rk4_ls", save_every=None):
    """Integrate y' = f(t, y) on ``t_span`` with ``n_steps`` fixed steps.

    Only the current state is kept in memory. With ``save_every=k`` the state is
    also copied every k steps (the initial and final states are always returned).
    ``f`` must return a new array: the steppers reuse it as scratch space.

    Returns ``(t, y)`` where ``y[j]`` is the state at time ``t[j]``.
    """
    t0, tf = t_span
    h = (tf - t0) / n_steps
    y = np.array(y0, dtype=float)
    step = METHODS[method](y.size)

    times, states = [t0], [y.copy()]
    for i in range(n_steps):
        step(f, t0 + i * h, y, h)
        if save_every and (i + 1) % save_every == 0 and i + 1 < n_steps:
            times.append(t0 + (i + 1) * h)
            states.append(y.copy())
    times.append(tf)
    states.append(y.copy())
    return np.array(times), np.array(states)
