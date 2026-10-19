"""Memory footprint and wall-clock time of each scheme as the dimension n grows.

Memory: tracemalloc peak while building the stepper and taking a few steps,
expressed in vectors of n floats (the state y itself is allocated beforehand and
not counted). Time: best of several runs, per step.
"""

import time
import tracemalloc

import numpy as np

from lsrk.problems import coupled_rhs, initial_state
from lsrk.schemes import METHODS
from style import COLORS, INK_2, LABELS, plt, save

MEMORY_N = 1_000_000
TIME_DIMS = [1_000, 10_000, 100_000, 1_000_000, 4_000_000]
TIMED = ["rk4", "rk4_ls", "rk4_ls_numexpr"]


def peak_vectors(method, n, steps=3):
    y = initial_state(n)
    tracemalloc.start()
    step = METHODS[method](n)
    for i in range(steps):
        step(coupled_rhs, i * 0.01, y, 0.01)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (8 * n)


def time_per_step(method, n, repeats=7):
    steps = max(10, min(200, 2_000_000 // n))
    best = np.inf
    for _ in range(repeats):
        y = initial_state(n)
        step = METHODS[method](n)
        start = time.perf_counter()
        for i in range(steps):
            step(coupled_rhs, i * 0.01, y, 0.01)
        best = min(best, (time.perf_counter() - start) / steps)
    return best


def main():
    memory = {m: peak_vectors(m, MEMORY_N) for m in METHODS}
    print(f"Peak extra memory at n = {MEMORY_N:,} (in vectors of n floats)")
    for m, v in memory.items():
        print(f"  {m:<16} {v:5.2f}")

    times = {m: [time_per_step(m, n) for n in TIME_DIMS] for m in TIMED}
    print("\nTime per step (ms)")
    print(f"  {'n':>10} " + " ".join(f"{m:>16}" for m in TIMED))
    for j, n in enumerate(TIME_DIMS):
        print(f"  {n:>10,} " + " ".join(f"{1e3 * times[m][j]:>16.3f}" for m in TIMED))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 1.2]})

    # Euler keeps no work array, so its peak is the cost of evaluating f alone
    order = ["euler", "rk2", "rk4", "rk4_ls", "rk4_ls_numexpr"]
    names = [LABELS[m] for m in order]
    f_cost = memory["euler"]
    own = [max(memory[m] - f_cost, 0.0) for m in order]
    ax1.barh(names, [f_cost] * len(order), color="#c3c2b7", height=0.6, label="evaluating f")
    bars = ax1.barh(names, own, left=f_cost, color=[COLORS[m] for m in order], height=0.6,
                    edgecolor="white", linewidth=2)
    ax1.bar_label(bars, labels=[f"+{v:.0f} n" for v in own], padding=4, color=INK_2)
    ax1.invert_yaxis()
    ax1.set_xlabel("peak extra memory (vectors of n floats)")
    ax1.set_title(f"Memory: integrator work arrays on top of f, n = {MEMORY_N:,}")
    ax1.grid(axis="y", visible=False)
    ax1.set_axisbelow(True)
    ax1.set_xlim(0, max(memory.values()) * 1.25)
    ax1.legend(loc="lower right")

    base = np.array(times["rk4"])
    ax2.axhline(1.0, color=COLORS["rk4"], linewidth=2, label=LABELS["rk4"] + " (reference)")
    for m in ["rk4_ls", "rk4_ls_numexpr"]:
        ax2.semilogx(TIME_DIMS, np.array(times[m]) / base, "o-", color=COLORS[m], label=LABELS[m])
    ax2.axhline(1.25, color=INK_2, linewidth=1, linestyle=":")
    ax2.text(1.4 * TIME_DIMS[0], 1.31, "5 stages / 4 stages = 1.25", color=INK_2, fontsize=9)
    ax2.set_xlabel("dimension n")
    ax2.set_ylabel("time per step / RK4 classic")
    ax2.set_title("Time per step, relative to RK4 classic")
    ax2.legend(loc="upper right")
    save(fig, "benchmark.png")


if __name__ == "__main__":
    main()
