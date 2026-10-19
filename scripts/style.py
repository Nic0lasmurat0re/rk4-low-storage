"""Shared figure style: one color per method, the same in every figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

COLORS = {
    "rk4": "#2a78d6",
    "rk4_ls": "#eb6834",
    "rk2": "#1baf7a",
    "euler": "#eda100",
    "rk4_ls_numexpr": "#6250d6",
}
LABELS = {
    "euler": "Euler",
    "rk2": "RK2 (Heun)",
    "rk4": "RK4 classic",
    "rk4_ls": "RK4 low-storage",
    "rk4_ls_numexpr": "RK4-LS + numexpr",
}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "legend.frameon": False,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "font.size": 10,
})


def save(fig, name):
    path = FIGURES / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"saved {path.relative_to(FIGURES.parent)}")
