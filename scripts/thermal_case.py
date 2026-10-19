"""Spacecraft nodal thermal model integrated over several LEO orbits.

Left: node temperatures with eclipse phases shaded (RK4 low-storage).
Right: accuracy against cost for RK4 classic and RK4 low-storage, with the
largest stable step of each scheme marked.
"""

import numpy as np
from scipy.integrate import solve_ivp

from lsrk.stability import real_axis_limit
from lsrk.thermal import random_model, simulate
from style import COLORS, INK_2, LABELS, plt, save

N_NODES, N_ORBITS, T0 = 20, 8, 250.0
STEPS_PER_ORBIT = [20, 25, 40, 50, 100, 200, 400]
STAGES = {"rk4": 4, "rk4_ls": 5}


def reference(model):
    T = np.full(model.n, T0)
    for start, end, sunlit in model.phases(N_ORBITS):
        T = solve_ivp(model.phase_rhs(sunlit), (start, end), T, method="DOP853",
                      rtol=1e-13, atol=1e-10).y[:, -1]
    return T


def main():
    model = random_model(N_NODES)
    P = model.orbit_period
    t, T = simulate(model, np.full(N_NODES, T0), N_ORBITS, 200, "rk4_ls", save_every=1)
    T_ref = reference(model)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1.5, 1]})

    for k in range(N_ORBITS):
        ax1.axvspan(k + model.sunlit_fraction, k + 1, color="#e4e3df", linewidth=0)
    hours = t / P
    celsius = T - 273.15
    for i in range(N_NODES):
        ax1.plot(hours, celsius[:, i], color=COLORS["rk4_ls"], linewidth=0.9, alpha=0.45)
    ax1.plot(hours, celsius.mean(axis=1), color="#0b0b0b", linewidth=2, label="mean over nodes")
    ax1.set_xlim(0, N_ORBITS)
    ax1.set_xlabel("orbits (shaded: eclipse)")
    ax1.set_ylabel("temperature (°C)")
    ax1.set_title(f"{N_NODES}-node spacecraft model, {N_ORBITS} LEO orbits of {P / 60:.0f} min")
    ax1.legend(loc="lower right")

    print(f"{'method':<8} {'steps/orbit':>11} {'h (s)':>7} {'max error (K)':>14}")
    for m in ["rk4", "rk4_ls"]:
        h_max = model.stable_step(real_axis_limit(m), safety=1.0)
        evals, errors = [], []
        for spo in STEPS_PER_ORBIT:
            Y = simulate(model, np.full(N_NODES, T0), N_ORBITS, spo, m)[1]
            err = np.max(np.abs(Y[-1] - T_ref))
            print(f"{m:<8} {spo:>11} {P / spo:>7.0f} {err:>14.2e}")
            evals.append(STAGES[m] * spo)
            errors.append(err)
        ax2.loglog(evals, errors, "o-", color=COLORS[m], label=LABELS[m])
        ax2.axvline(STAGES[m] * P / h_max, color=COLORS[m], linestyle=":", linewidth=1.5)
        print(f"{m:<8} guaranteed-stable step h <= {h_max:.0f} s ({P / h_max:.0f} steps/orbit)")
    ax2.text(0.98, 0.98, "dotted: cost at the guaranteed\nstability limit (Gershgorin)",
             transform=ax2.transAxes, ha="right", va="top", color=INK_2, fontsize=9)
    ax2.set_xlabel("evaluations of f per orbit")
    ax2.set_ylabel("max error after 8 orbits (K)")
    ax2.set_title("Accuracy against cost")
    ax2.legend(loc="lower left")
    save(fig, "thermal.png")


if __name__ == "__main__":
    main()
