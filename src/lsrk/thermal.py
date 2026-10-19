"""Nodal thermal model of a spacecraft in low Earth orbit.

Each node i is isothermal, with heat capacity MC_i, and obeys

    MC_i dT_i/dt = P_i(t)                                   absorbed fluxes + dissipation
                   + sum_j GL_ij (T_j - T_i)                 conduction to neighbours
                   + sigma sum_j GR_ij (T_j^4 - T_i^4)        radiation to neighbours
                   - eps_i S_i sigma T_i^4                   emission to deep space

with P_i(t) = alpha_i S_i (phi_S + phi_albedo) s(t) + eps_i S_i phi_IR + Q_i and
s(t) = 1 in sunlight, 0 in eclipse. The nodes form a chain (tridiagonal coupling),
stored sparse so that the model scales to large n.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant, W/m^2/K^4
PHI_SUN = 1361.0  # solar constant, W/m^2
ALBEDO = 0.30
PHI_ALBEDO = ALBEDO * PHI_SUN * 0.30  # albedo flux reaching the spacecraft (view factor 0.3)
PHI_IR = 237.0  # Earth infrared flux, W/m^2


@dataclass
class NodalModel:
    MC: np.ndarray  # heat capacity of each node, J/K
    area: np.ndarray  # radiating area, m^2
    alpha: np.ndarray  # solar absorptivity
    eps: np.ndarray  # infrared emissivity
    Q: np.ndarray  # internal dissipation, W
    GL: sp.csr_matrix  # conductive Laplacian, W/K (GL_ij on off-diagonal, -row sum on diagonal)
    GR: sp.csr_matrix  # radiative Laplacian, m^2 (same structure)
    orbit_period: float = 5400.0  # s
    sunlit_fraction: float = 0.6

    @property
    def n(self):
        return self.MC.size

    def sunlit(self, t):
        return (t % self.orbit_period) < self.sunlit_fraction * self.orbit_period

    def rhs(self, t, T):
        """dT/dt as a new array, with the sunlit/eclipse state taken from t."""
        return self._rhs(T, self.sunlit(t))

    def phase_rhs(self, sunlit):
        """dT/dt with the sunlit/eclipse state fixed, for integrating one phase at a time."""
        return lambda t, T: self._rhs(T, sunlit)

    def _rhs(self, T, sunlit):
        T4 = T**4
        q = self.GL @ T
        q += SIGMA * (self.GR @ T4)
        q -= SIGMA * self.eps * self.area * T4
        q += self.eps * self.area * PHI_IR + self.Q
        if sunlit:
            q += self.alpha * self.area * (PHI_SUN + PHI_ALBEDO)
        q /= self.MC
        return q

    def phases(self, n_orbits):
        """(start, end, sunlit) of each sunlit and eclipse phase over n_orbits."""
        P, t_sun = self.orbit_period, self.sunlit_fraction * self.orbit_period
        for k in range(n_orbits):
            yield k * P, k * P + t_sun, True
            yield k * P + t_sun, (k + 1) * P, False

    def spectral_radius_bound(self, T_max):
        """Gershgorin bound on the spectral radius of the Jacobian of rhs for T <= T_max."""
        GL = abs(self.GL).sum(axis=1).A1  # = 2 * conductance row sum
        GR = abs(self.GR).sum(axis=1).A1
        radiative = 4 * SIGMA * T_max**3 * (GR + self.eps * self.area)
        return np.max((GL + radiative) / self.MC)

    def stable_step(self, stability_limit, T_max=450.0, safety=0.8):
        """Largest step allowed by linear stability, with a safety factor."""
        return safety * stability_limit / self.spectral_radius_bound(T_max)


def simulate(model, T0, n_orbits, steps_per_orbit, method="rk4_ls", save_every=None):
    """Integrate the model phase by phase with a fixed step.

    The solar flux jumps at each sunlit/eclipse transition. Restarting the integrator
    at every transition keeps each step on a smooth right-hand side; otherwise a stage
    evaluated exactly on the jump (t + h for classical RK4) sees the wrong phase and the
    scheme drops to first order.
    """
    from .schemes import solve

    steps_sun = steps_per_orbit * model.sunlit_fraction
    if abs(steps_sun - round(steps_sun)) > 1e-9:
        raise ValueError("steps_per_orbit * sunlit_fraction must be an integer")
    steps_sun = round(steps_sun)
    steps = {True: steps_sun, False: steps_per_orbit - steps_sun}

    times, states, T = [], [], np.array(T0, dtype=float)
    for start, end, sunlit in model.phases(n_orbits):
        t, Y = solve(model.phase_rhs(sunlit), (start, end), T, steps[sunlit], method, save_every)
        times.append(t[:-1])
        states.append(Y[:-1])
        T = Y[-1]
    times.append([end])
    states.append([T])
    return np.concatenate(times), np.concatenate(states)


def chain_laplacian(g):
    """Laplacian of a chain whose link i connects nodes i and i+1 with weight g[i]."""
    n = g.size + 1
    diag = np.zeros(n)
    diag[:-1] -= g
    diag[1:] -= g
    return sp.diags([g, diag, g], [-1, 0, 1], format="csr")


def random_model(n=20, seed=42):
    """A chain of n nodes with physically plausible random properties."""
    rng = np.random.default_rng(seed)
    return NodalModel(
        MC=rng.uniform(500.0, 2000.0, n),
        area=rng.uniform(0.01, 0.10, n),
        alpha=rng.uniform(0.2, 0.9, n),
        eps=rng.uniform(0.5, 0.95, n),
        Q=rng.uniform(0.0, 5.0, n),
        GL=chain_laplacian(rng.uniform(0.1, 2.0, n - 1)),
        GR=chain_laplacian(rng.uniform(1e-4, 5e-3, n - 1)),
    )
