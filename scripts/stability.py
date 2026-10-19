"""Linear stability regions |R(z)| <= 1 and real-axis limits of each scheme."""

import numpy as np

from lsrk.stability import real_axis_limit, stability_polynomial
from style import COLORS, LABELS, plt, save

METHODS = ["euler", "rk2", "rk4", "rk4_ls"]
STAGES = {"euler": 1, "rk2": 2, "rk4": 4, "rk4_ls": 5}


def main():
    x, y = np.meshgrid(np.linspace(-5.5, 1.0, 600), np.linspace(-4.0, 4.0, 600))
    z = x + 1j * y

    fig, ax = plt.subplots(figsize=(7, 5))
    print(f"{'method':<10} {'real limit':>11} {'limit / stage':>14}")
    for m in METHODS:
        R = stability_polynomial(m)
        r = real_axis_limit(m)
        print(f"{m:<10} {r:>11.3f} {r / STAGES[m]:>14.3f}")
        ax.contour(x, y, np.abs(R(z)), levels=[1.0], colors=COLORS[m], linewidths=2)
        ax.plot([], [], color=COLORS[m], label=f"{LABELS[m]}: real limit {r:.2f}, per stage {r / STAGES[m]:.2f}")

    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.axvline(0, color="#52514e", linewidth=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("Re(h λ)")
    ax.set_ylabel("Im(h λ)")
    ax.set_title("Stability regions |R(hλ)| ≤ 1")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=9)
    save(fig, "stability.png")


if __name__ == "__main__":
    main()
