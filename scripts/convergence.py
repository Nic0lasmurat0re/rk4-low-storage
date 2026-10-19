"""Order of convergence of each scheme, measured against a tight reference solution.

Errors are plotted against the number of evaluations of f, the honest cost measure
when comparing schemes with a different number of stages.
"""

import numpy as np
from scipy.integrate import solve_ivp

from lsrk import solve
from lsrk.problems import coupled_rhs, initial_state
from style import COLORS, INK_2, LABELS, plt, save

METHODS = ["euler", "rk2", "rk4", "rk4_ls"]
STAGES = {"euler": 1, "rk2": 2, "rk4": 4, "rk4_ls": 5}
N_DIM, T_SPAN = 10, (0.0, 10.0)
STEPS = [25, 50, 100, 200, 400, 800, 1600]


def main():
    y0 = initial_state(N_DIM)
    ref = solve_ivp(coupled_rhs, T_SPAN, y0, method="DOP853", rtol=1e-13, atol=1e-13).y[:, -1]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    print(f"{'method':<10} {'observed order':>15}")
    for m in METHODS:
        errors = np.array([np.max(np.abs(solve(coupled_rhs, T_SPAN, y0, n, m)[1][-1] - ref)) for n in STEPS])
        keep = errors > 1e-13  # below this the reference itself is the limit
        evals = STAGES[m] * np.array(STEPS)
        slope = -np.polyfit(np.log(np.array(STEPS)[keep]), np.log(errors[keep]), 1)[0]
        print(f"{m:<10} {slope:>15.2f}")
        ax.loglog(evals[keep], errors[keep], "o-", color=COLORS[m], label=f"{LABELS[m]} (order {slope:.2f})")

    ax.set_xlabel("evaluations of f")
    ax.set_ylabel("max error at t = 10")
    ax.set_title("Convergence on the coupled test system (n = 10)")
    ax.legend(loc="lower left")
    ax.text(0.98, 0.98, "lower and further left is better", transform=ax.transAxes,
            ha="right", va="top", color=INK_2, fontsize=9)
    save(fig, "convergence.png")


if __name__ == "__main__":
    main()
