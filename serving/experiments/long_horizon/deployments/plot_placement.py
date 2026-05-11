#!/usr/bin/env python3
"""Plot placed-task capacity over time from the deployment plan.

Shows three lines:
  - Active tasks (demand) — how many tasks exist at each moment
  - FMaaS supported tasks — matches demand (all placed)
  - No-Sharing supported tasks — saturates when memory runs out

Usage (from serving/):
    python -m experiments.long_horizon.deployments.plot_placement
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

HERE = Path(__file__).resolve().parent


def _step_count(intervals, duration, t_step=1.0):
    """Count how many intervals [arrive, depart) are active at each t."""
    t = np.arange(0.0, duration + t_step, t_step)
    counts = np.zeros(len(t))
    for a, d in intervals:
        counts += ((t >= a) & (t < d)).astype(float)
    return t, counts


def main():
    tl = json.loads((HERE / "task_timeline.json").read_text())
    duration = 1800.0

    # ── Demand: all tasks from timeline ──
    demand_ivs = [(float(v["arrive"]), float(v["depart"])) for v in tl.values()]

    # ── Placed tasks per condition ──
    cond_ivs = {}
    for cond, label in [("no_sharing", "No-Sharing"), ("fmaas", "FMaaS")]:
        path = HERE / f"{cond}_slots.json"
        if not path.is_file():
            print(f"[plot] {path} not found, skipping {cond}")
            continue
        slots = json.loads(path.read_text())

        placed_tasks = set()
        if cond == "fmaas":
            # Each slot groups tasks under a backbone
            for s in slots:
                for t in s.get("tasks", []):
                    placed_tasks.add(t["task"] if isinstance(t, dict) else t)
        else:
            # Each slot is one task
            for s in slots:
                if "task" in s:
                    placed_tasks.add(s["task"])

        ivs = []
        for task_name in placed_tasks:
            if task_name in tl:
                ivs.append((float(tl[task_name]["arrive"]),
                            float(tl[task_name]["depart"])))
        cond_ivs[cond] = (ivs, placed_tasks, label)

    # ── Plot ──
    COLORS = {"fmaas": "#2166ac", "no_sharing": "#d6604d"}

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "black", "axes.spines.top": False,
        "axes.spines.right": False, "font.family": "sans-serif",
        "font.size": 10, "axes.labelsize": 11,
        "pdf.fonttype": 42, "savefig.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(8, 3.5))

    # Demand line
    t, dem = _step_count(demand_ivs, duration)
    ax.step(t, dem, where="post", color="black", lw=1.3, ls="--",
            alpha=0.7, label="Active tasks (demand)", zorder=4)
    ax.fill_between(t, 0, dem, step="post", alpha=0.05, color="black")

    # Per-condition lines
    for cond in ["fmaas", "no_sharing"]:
        if cond not in cond_ivs:
            continue
        ivs, placed, label = cond_ivs[cond]
        t_c, cnt = _step_count(ivs, duration)
        ax.step(t_c, cnt, where="post", color=COLORS[cond], lw=2.0,
                label=f"{label} ({len(placed)} placed)", zorder=3)

        # Shade gap for no_sharing
        if cond == "no_sharing":
            ax.fill_between(t, cnt[:len(t)], dem,
                            where=dem > cnt[:len(t)], step="post",
                            alpha=0.12, color=COLORS[cond],
                            label="Unplaced gap")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Number of tasks")
    ax.set_xlim(0, duration)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        p = HERE / f"placement_capacity.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150 if ext == "png" else 300)
        print(f"[plot] wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
