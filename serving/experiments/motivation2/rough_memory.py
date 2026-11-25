#!/usr/bin/env python3
"""motivation2/rough_memory.py — a rough figure (not for the paper): where the
GPU's memory goes in each isolation run, one panel per condition.

Per panel, from isolation/mem_<condition>.jsonl (Control("stats") every 0.5 s):
    noisy B     B's memory, as plot.py's panel (c) charges it
    victim      the same rule applied to the victim task(s), summed
    other       device used - B - victim: weights, decoders, activations,
                CUDA contexts, allocator cache not charged to a task
    device used the GPU's used memory as CUDA reports it (contexts
                included); for mig, the slices' used memory summed
with the GPU's capacity (and, for mig, one slice) dashed.

Usage (from serving/):
    python experiments/motivation2/rough_memory.py --exp-dir experiments/motivation2/results_swinlarge
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("m2plot", HERE / "plot.py")
pl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pl)

CONDS = ["shared", "shared_bfq", "fmvisor", "mig", "no_sharing"]
GRID_S = 0.5


def on_grid(t, y, grid):
    if not len(t):
        return np.full_like(grid, np.nan)
    return np.interp(grid, t, y, left=np.nan, right=np.nan)


def device_used(rows: list, s: dict, grid) -> np.ndarray:
    """GB the GPU has in use: one server's view, or the MIG slices' summed."""
    by_url: dict = {}
    for r in rows:
        if r.get("device_used_mb") is not None:
            by_url.setdefault(r["url"], []).append((r["t"], r["device_used_mb"]))
    if not by_url:
        return np.full_like(grid, np.nan)
    curves = []
    for pts in by_url.values():
        pts.sort()
        curves.append(on_grid(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]), grid))
    if s["condition"] == "mig":
        return np.nansum(curves, axis=0) / 1024
    return np.nanmax(curves, axis=0) / 1024     # one GPU: every server sees the same


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    d = a.exp_dir / "isolation"
    summ = pl.latest_summaries(d / "summary.jsonl")
    conds = [c for c in CONDS if c in summ and (d / f"mem_{c}.jsonl").exists()]
    if not conds:
        raise SystemExit(f"no mem_<condition>.jsonl in {d}")

    fig, axes = plt.subplots(1, len(conds), figsize=(3.0 * len(conds), 2.4),
                             sharey=True, squeeze=False)
    for ax, c in zip(axes[0], conds):
        s = summ[c]
        rows = pl.load_stats(d / f"mem_{c}.jsonl")
        dur = s["duration"]
        grid = np.arange(0.0, dur, GRID_S)
        tb, yb, quota = pl.noisy_memory_from_stats(rows, s)
        b = on_grid(tb, yb, grid)
        v = np.zeros_like(grid)
        for task in s["victims"]:
            tv, yv, _ = pl.noisy_memory_from_stats(rows, s, task)
            v = v + np.nan_to_num(on_grid(tv, yv, grid))
        dev = device_used(rows, s, grid)
        other = np.clip(dev - np.nan_to_num(b) - v, 0, None)

        ax.plot(grid, b, color="#c0392b", lw=1.4, label="noisy B")
        ax.plot(grid, v, color="#2471a3", lw=1.4, label="victim")
        ax.plot(grid, other, color="#7f8c8d", lw=1.0, ls="--", label="other (weights, ctx, cache)")
        ax.plot(grid, dev, color="black", lw=0.9, label="device used")
        cap = (s.get("gpu_total_mb") or 0) / 1024
        if cap:
            ax.axhline(cap, color="#999999", lw=0.6, ls=":")
            ax.text(2, cap, "GPU capacity", fontsize=6, color="#777777", va="bottom")
        if c == "mig" and s.get("mig_slice_mb"):
            sl = s["mig_slice_mb"] / 1024
            ax.axhline(sl, color="#2E8B57", lw=0.6, ls=":")
            ax.text(2, sl, "one slice", fontsize=6, color="#2E8B57", va="bottom")
        if quota:
            ax.axhline(quota, color="#c0392b", lw=0.6, ls=":")
            ax.text(2, quota, "B's quota", fontsize=6, color="#c0392b", va="bottom")
        ax.axvline(s["ramp_at"], color="black", lw=0.4, ls=":")
        ax.set_title(pl.LABELS.get(c, c) if c != "mig" else "Partitioned (MIG)", fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_xlim(0, dur)
        ax.tick_params(labelsize=6.5)
        ax.grid(True, color="#eeeeee", lw=0.4)
    axes[0][0].set_ylabel("GPU memory (GB)", fontsize=7)
    axes[0][0].legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.92))
    fig.tight_layout()
    out = a.out or a.exp_dir / "rough_memory"
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
