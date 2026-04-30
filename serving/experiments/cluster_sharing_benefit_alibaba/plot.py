#!/usr/bin/env python3
"""Plots for the cluster_sharing_benefit sweep.

Reads results/N{N}/<condition>/request_latency_results.csv and produces, in
results/plots/:

Sweep-level (across N):
    latency_vs_napps_mean.pdf  / _p95.pdf / _p99.pdf   — grouped bars per N
    throughput_vs_napps.pdf                            — grouped bars + offered-load
    latency_cdf_N{N}.pdf                               — one CDF per N

Per-N detail (one set per N that has data):
    perN/N{N}/per_task_p50_latency.pdf / p95 / p99
    perN/N{N}/per_task_service_time.pdf
    perN/N{N}/summary_bars.pdf

Usage:
    python experiments/cluster_sharing_benefit/plot.py \\
        [--exp-dir experiments/cluster_sharing_benefit/results] \\
        [--warmup-secs 10]
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

SERVING_DIR = Path(__file__).resolve().parents[2]
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd


# ── Style ──────────────────────────────────────────────────────────

SERIES_ORDER  = ["no_sharing_tpc", "no_sharing", "sharing"]
SERIES_COLORS = {
    "no_sharing_tpc": "#6B9AC4",
    "no_sharing":     "#888888",
    "sharing":        "#E06C75",
}
SERIES_LABELS = {
    "no_sharing":     "NS",
    "no_sharing_tpc": "NS (TPC)",
    "sharing":        "FMVisor",
}
SERIES_LINESTYLE = {
    "no_sharing_tpc": "--",
    "no_sharing":     "-.",
    "sharing":        "-",
}

USECOLS = [
    "req_time",
    "end_to_end_latency(ms)",
    "backend_exec_time(ms)",
    "proc_time(ms)",
    "decoder_time(ms)",
    "device",
    "device_start_time",
    "task",
    "backbone",
]
DTYPES = {
    "req_time":               "float32",
    "end_to_end_latency(ms)": "float32",
    "backend_exec_time(ms)":  "float32",
    "proc_time(ms)":          "float32",
    "decoder_time(ms)":       "float32",
    "device":                 "category",
    "device_start_time":      "float64",
    "task":                   "category",
    "backbone":               "category",
}


def apply_paper_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "black",
        "axes.labelcolor":   "black",
        "axes.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "grid.color":        "#cccccc",
        "grid.linestyle":    ":",
        "grid.linewidth":    0.4,
        "xtick.color":       "black",
        "ytick.color":       "black",
        "font.family":       "sans-serif",
        "font.size":         8,
        "axes.titlesize":    9,
        "axes.labelsize":    8,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.fontsize":   7,
        "lines.linewidth":   1.3,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "savefig.facecolor": "white",
        "savefig.bbox":      "tight",
    })


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[Plot] Saved: {out_path}")


def _plain_number_axis(ax: plt.Axes, axis: str = "y") -> None:
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    elif axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    else:
        raise ValueError(axis)


def _series_is_all_zero_or_nan(vals: List[float]) -> bool:
    if not vals:
        return True
    arr = np.asarray(vals, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return True
    return bool(np.allclose(finite, 0.0))


# ── Loading ────────────────────────────────────────────────────────

N_DIR_RE = re.compile(r"^N(\d+)$")
RUN_DONE_RE = re.compile(r"Done\. (?P<success>\d+) successful / (?P<total>\d+) total requests\.")


def _load_df(csv_path: Path, warmup_secs: float) -> pd.DataFrame | None:
    """Load only the columns we need, in chunks, already warmup-filtered."""
    if not csv_path.is_file():
        return None
    try:
        size_gb = csv_path.stat().st_size / 1e9
        print(f"[Plot] Loading {csv_path} ({size_gb:.1f} GB) ...", flush=True)
        chunks: List[pd.DataFrame] = []
        for chunk in pd.read_csv(
            csv_path,
            usecols=USECOLS,
            dtype=DTYPES,
            engine="c",
            chunksize=2_000_000,
            on_bad_lines="skip",
        ):
            chunk = chunk[chunk["req_time"] >= warmup_secs]
            if not chunk.empty:
                chunks.append(chunk)
        if not chunks:
            return None
        df = pd.concat(chunks, ignore_index=True, copy=False)
        return df
    except Exception as e:
        print(f"[Plot] Failed to load {csv_path}: {e}")
        return None


def _scan(exp_dir: Path, warmup_secs: float
          ) -> Dict[str, Dict[int, pd.DataFrame]]:
    """Return {condition -> {N -> DataFrame}}."""
    out: Dict[str, Dict[int, pd.DataFrame]] = {}
    for n_dir in sorted(exp_dir.iterdir()):
        m = N_DIR_RE.match(n_dir.name) if n_dir.is_dir() else None
        if not m:
            continue
        n = int(m.group(1))
        for cond_dir in sorted(n_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            df = _load_df(cond_dir / "request_latency_results.csv", warmup_secs)
            if df is None or df.empty:
                continue
            out.setdefault(cond_dir.name, {})[n] = df
            print(f"[Plot] N={n} {cond_dir.name}: {len(df):,} requests")
    return out


def _scan_run_status(exp_dir: Path) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Return {condition -> {N -> {success,total,failed,success_rate}}} from run.log."""
    out: Dict[str, Dict[int, Dict[str, float]]] = {}
    for n_dir in sorted(exp_dir.iterdir()):
        m = N_DIR_RE.match(n_dir.name) if n_dir.is_dir() else None
        if not m:
            continue
        n = int(m.group(1))
        for cond_dir in sorted(n_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            log_path = cond_dir / "run.log"
            if not log_path.is_file():
                continue
            text = log_path.read_text(errors="ignore")
            match = RUN_DONE_RE.search(text)
            if not match:
                continue
            success = float(match.group("success"))
            total = float(match.group("total"))
            failed = max(0.0, total - success)
            success_rate = 100.0 * success / total if total > 0 else 0.0
            out.setdefault(cond_dir.name, {})[n] = {
                "success": success,
                "total": total,
                "failed": failed,
                "success_rate": success_rate,
            }
    return out


# ── Stats ──────────────────────────────────────────────────────────

def _stat_arr(a: np.ndarray, kind: str) -> float:
    if a.size == 0:
        return 0.0
    if kind == "mean": return float(a.mean())
    if kind == "p50":  return float(np.percentile(a, 50))
    if kind == "p95":  return float(np.percentile(a, 95))
    if kind == "p99":  return float(np.percentile(a, 99))
    raise ValueError(kind)


def _lat_arr(df: pd.DataFrame) -> np.ndarray:
    return df["end_to_end_latency(ms)"].to_numpy()


def _exec_arr(df: pd.DataFrame) -> np.ndarray:
    return df["backend_exec_time(ms)"].to_numpy()


def _throughput(df: pd.DataFrame, warmup_secs: float) -> float:
    if df.empty:
        return 0.0
    req_times = df["req_time"].to_numpy()
    lat_s = _lat_arr(df) / 1000.0
    comp_times = req_times + lat_s - warmup_secs
    end = float(req_times.max()) - warmup_secs
    if end <= 0:
        return 0.0
    n_bins = int(np.ceil(end))
    if n_bins <= 0:
        return 0.0
    idx = comp_times.astype(np.int64)
    idx = idx[(idx >= 0) & (idx < n_bins)]
    counts = np.bincount(idx, minlength=n_bins).astype(float)
    if counts.size == 0:
        return 0.0
    return float(counts.mean())


# ── Sweep plots ────────────────────────────────────────────────────

def _lines_vs_n(data: Dict[str, Dict[int, pd.DataFrame]],
                value_fn, ylabel: str, out_path: Path,
                ylim: Tuple[float, float] | None = None) -> None:
    if not data:
        return
    all_ns = sorted({n for d in data.values() for n in d.keys()})
    keys = [k for k in SERIES_ORDER if k in data]
    if not all_ns or not keys:
        return
    fig, ax = plt.subplots(figsize=(max(4.4, 0.65 * len(all_ns) + 1.4), 2.8))
    for key in keys:
        ys = [
            value_fn(data[key][n]) if n in data[key] else np.nan
            for n in all_ns
        ]
        if _series_is_all_zero_or_nan(ys):
            continue
        ax.plot(all_ns, ys,
                color=SERIES_COLORS[key],
                linestyle=SERIES_LINESTYLE[key],
                marker="o",
                markersize=4.0,
                markeredgewidth=0.0,
                solid_capstyle="round",
                solid_joinstyle="round",
                antialiased=True,
                label=SERIES_LABELS[key])
    ax.set_xticks(all_ns)
    ax.set_xlabel("Number of applications")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(bottom=0.0)
    ax.margins(x=0.04)
    ax.grid(True, axis="y")
    _plain_number_axis(ax, "y")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_latency_vs_n(data, out_dir: Path) -> None:
    for stat, label in [("mean", "Mean E2E latency (ms)"),
                        ("p95",  "p95 E2E latency (ms)"),
                        ("p99",  "p99 E2E latency (ms)")]:
        _lines_vs_n(data,
                    lambda df, s=stat: _stat_arr(_lat_arr(df), s),
                    label, out_dir / f"latency_vs_napps_{stat}.pdf",
                    ylim=(0.0, 300.0))


def _bars_vs_n(data, value_fn, ylabel: str, out_path: Path) -> None:
    if not data:
        return
    all_ns = sorted({n for d in data.values() for n in d.keys()})
    keys = [k for k in SERIES_ORDER if k in data]
    if not all_ns or not keys:
        return

    fig, ax = plt.subplots(figsize=(1.8, 1.2))
    x = np.arange(len(all_ns), dtype=float)
    width = 0.8 / len(keys)
    vmax = 0.0
    for i, key in enumerate(keys):
        ys = [
            value_fn(data[key][n]) if n in data[key] else 0.0
            for n in all_ns
        ]
        if _series_is_all_zero_or_nan(ys):
            continue
        bars = ax.bar(
            x + (i - (len(keys) - 1) / 2) * width, ys, width,
            color=SERIES_COLORS[key], edgecolor="black", linewidth=0.4,
            label=SERIES_LABELS[key],
        )
        for b, v in zip(bars, ys):
            if v > 0 and np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{v:.0f}", ha="center", va="bottom", fontsize=5.0)
        vmax = max(vmax, max((v for v in ys if np.isfinite(v)), default=0.0))
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in all_ns])
    ax.set_xlabel("N (apps)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, _nice_upper(vmax * 1.12))
    ax.grid(True, axis="y")
    ax.legend(frameon=False, ncol=min(3, len(keys)),
              handlelength=1.0, handletextpad=0.3, columnspacing=0.7,
              loc="lower center", bbox_to_anchor=(0.5, 1.02),
              borderaxespad=0.0)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_latency_bars_vs_n(data, out_dir: Path) -> None:
    for stat, label in [("mean", "Mean RT (ms)"),
                        ("p95",  "p95 RT (ms)"),
                        ("p99",  "p99 RT (ms)")]:
        _bars_vs_n(data,
                   lambda df, s=stat: _stat_arr(_lat_arr(df), s),
                   label, out_dir / f"latency_bars_{stat}.pdf")


# ── Memory (from model_deployment_results.json) ────────────────────

def _gpu_peak_mb_for_deployment(entry: dict) -> float:
    """Sum gpu peak (MB) across load_backbone + add_decoder_* in one entry."""
    raw = entry.get("logger_summary")
    if not raw:
        return 0.0
    try:
        d = ast.literal_eval(raw) if isinstance(raw, str) else raw
    except Exception:
        return 0.0
    total = 0.0
    for k, v in d.items():
        if not (k == "load_backbone" or k.startswith("add_decoder_")):
            continue
        if isinstance(v, dict):
            total += float(v.get("gpu peak", 0.0) or 0.0)
    return total


def _total_gpu_peak_mb(cond_dir: Path) -> float | None:
    p = cond_dir / "model_deployment_results.json"
    if not p.is_file():
        return None
    try:
        entries = json.loads(p.read_text())
    except Exception:
        return None
    if not isinstance(entries, list):
        return None
    total = sum(_gpu_peak_mb_for_deployment(e) for e in entries)
    return float(total)


def _scan_gpu_peak(exp_dir: Path) -> Dict[str, Dict[int, float]]:
    out: Dict[str, Dict[int, float]] = {}
    for n_dir in sorted(exp_dir.iterdir()):
        m = N_DIR_RE.match(n_dir.name) if n_dir.is_dir() else None
        if not m:
            continue
        n = int(m.group(1))
        for cond_dir in sorted(n_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            mb = _total_gpu_peak_mb(cond_dir)
            if mb is None:
                continue
            out.setdefault(cond_dir.name, {})[n] = mb
    return out


def plot_memory_bars_vs_n(exp_dir: Path, out_path: Path,
                          ns_filter: set[int] | None = None) -> None:
    mem = _scan_gpu_peak(exp_dir)
    if not mem:
        return
    if ns_filter is not None:
        mem = {k: {n: v for n, v in d.items() if n in ns_filter}
               for k, d in mem.items()}
        mem = {k: d for k, d in mem.items() if d}
    all_ns = sorted({n for d in mem.values() for n in d.keys()})
    keys = [k for k in SERIES_ORDER if k in mem]
    if not all_ns or not keys:
        return
    # Use MB if small, GB if large.
    max_mb = max((mem[k].get(n, 0.0) for k in keys for n in all_ns), default=0.0)
    use_gb = max_mb >= 1024.0
    unit = "GB" if use_gb else "MB"
    scale = 1.0 / 1024.0 if use_gb else 1.0

    fig, ax = plt.subplots(figsize=(1.8, 1.2))
    x = np.arange(len(all_ns), dtype=float)
    width = 0.8 / len(keys)
    vmax = 0.0
    for i, key in enumerate(keys):
        ys = [mem[key].get(n, 0.0) * scale for n in all_ns]
        bars = ax.bar(
            x + (i - (len(keys) - 1) / 2) * width, ys, width,
            color=SERIES_COLORS[key], edgecolor="black", linewidth=0.4,
            label=SERIES_LABELS[key],
        )
        for b, v in zip(bars, ys):
            if v > 0 and np.isfinite(v):
                fmt = f"{v:.1f}" if use_gb else f"{v:.0f}"
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        fmt, ha="center", va="bottom", fontsize=5.0)
        vmax = max(vmax, max(ys, default=0.0))
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in all_ns])
    ax.set_xlabel("N (apps)")
    ax.set_ylabel(f"Memory ({unit})")
    ax.set_ylim(0, _nice_upper(vmax * 1.12))
    ax.grid(True, axis="y")
    ax.legend(frameon=False, ncol=min(3, len(keys)),
              handlelength=1.0, handletextpad=0.3, columnspacing=0.7,
              loc="lower center", bbox_to_anchor=(0.5, 1.02),
              borderaxespad=0.0)
    save_figure(fig, out_path)
    plt.close(fig)


# ── Runtime memory (per-deployment peak, summed) ───────────────────

def _runtime_peak_mb_sum(csv_path: Path) -> float | None:
    """Sum of per-deployment peak gpu_alloc_peak_mb across all deployments in a run.

    A deployment is identified by (device, backbone); peak is max over its requests.
    """
    if not csv_path.is_file():
        return None
    try:
        df = pd.read_csv(csv_path, usecols=["device", "backbone", "gpu_alloc_peak_mb"])
    except Exception:
        return None
    if df.empty or "gpu_alloc_peak_mb" not in df.columns:
        return None
    df = df.dropna(subset=["gpu_alloc_peak_mb"])
    if df.empty:
        return None
    per_dep_peak = df.groupby(["device", "backbone"])["gpu_alloc_peak_mb"].max()
    return float(per_dep_peak.sum())


def _scan_runtime_peak(exp_dir: Path) -> Dict[str, Dict[int, float]]:
    out: Dict[str, Dict[int, float]] = {}
    for n_dir in sorted(exp_dir.iterdir()):
        m = N_DIR_RE.match(n_dir.name) if n_dir.is_dir() else None
        if not m:
            continue
        n = int(m.group(1))
        for cond_dir in sorted(n_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            mb = _runtime_peak_mb_sum(cond_dir / "request_latency_results.csv")
            if mb is None:
                continue
            out.setdefault(cond_dir.name, {})[n] = mb
    return out


def plot_runtime_memory_bars_vs_n(exp_dir: Path, out_path: Path,
                                  ns_filter: set[int] | None = None) -> None:
    """Bar plot of sum-of-per-deployment runtime peak GPU memory vs N.

    Per-deployment peak = max(gpu_alloc_peak_mb) over requests for (device, backbone).
    Cluster total = sum across deployments; shared-weight bytes counted once per
    deployment, avoiding the per-request overcount.
    """
    mem = _scan_runtime_peak(exp_dir)
    if not mem:
        return
    if ns_filter is not None:
        mem = {k: {n: v for n, v in d.items() if n in ns_filter}
               for k, d in mem.items()}
        mem = {k: d for k, d in mem.items() if d}
    all_ns = sorted({n for d in mem.values() for n in d.keys()})
    keys = [k for k in SERIES_ORDER if k in mem]
    if not all_ns or not keys:
        return
    max_mb = max((mem[k].get(n, 0.0) for k in keys for n in all_ns), default=0.0)
    use_gb = max_mb >= 1024.0
    unit = "GB" if use_gb else "MB"
    scale = 1.0 / 1024.0 if use_gb else 1.0

    fig, ax = plt.subplots(figsize=(1.8, 1.2))
    x = np.arange(len(all_ns), dtype=float)
    width = 0.8 / len(keys)
    vmax = 0.0
    for i, key in enumerate(keys):
        ys = [mem[key].get(n, 0.0) * scale for n in all_ns]
        bars = ax.bar(
            x + (i - (len(keys) - 1) / 2) * width, ys, width,
            color=SERIES_COLORS[key], edgecolor="black", linewidth=0.4,
            label=SERIES_LABELS[key],
        )
        for b, v in zip(bars, ys):
            if v > 0 and np.isfinite(v):
                fmt = f"{v:.1f}" if use_gb else f"{v:.0f}"
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        fmt, ha="center", va="bottom", fontsize=5.0)
        vmax = max(vmax, max(ys, default=0.0))
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in all_ns])
    ax.set_xlabel("N (apps)")
    ax.set_ylabel(f"Runtime memory ({unit})")
    ax.set_ylim(0, _nice_upper(vmax * 1.12))
    ax.grid(True, axis="y")
    ax.legend(frameon=False, ncol=min(3, len(keys)),
              handlelength=1.0, handletextpad=0.3, columnspacing=0.7,
              loc="lower center", bbox_to_anchor=(0.5, 1.02),
              borderaxespad=0.0)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_throughput_vs_n(data, out_path: Path, warmup_secs: float) -> None:
    if not data:
        return
    all_ns = sorted({n for d in data.values() for n in d.keys()})
    keys = [k for k in SERIES_ORDER if k in data]
    if not all_ns or not keys:
        return
    fig, ax = plt.subplots(figsize=(max(4.4, 0.65 * len(all_ns) + 1.4), 2.8))
    for key in keys:
        ys = [
            _throughput(data[key][n], warmup_secs) if n in data[key] else np.nan
            for n in all_ns
        ]
        if _series_is_all_zero_or_nan(ys):
            continue
        ax.plot(all_ns, ys,
                color=SERIES_COLORS[key],
                linestyle=SERIES_LINESTYLE[key],
                marker="o",
                markersize=4.0,
                markeredgewidth=0.0,
                solid_capstyle="round",
                solid_joinstyle="round",
                antialiased=True,
                label=SERIES_LABELS[key])
    ax.set_xticks(all_ns)
    ax.set_xlabel("Number of applications")
    ax.set_ylabel("Completed throughput (req/s)")
    ax.set_ylim(bottom=0.0)
    ax.margins(x=0.04)
    ax.grid(True, axis="y")
    _plain_number_axis(ax, "y")
    ax.legend(frameon=False, ncol=2, fontsize=7)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_latency_cdfs(data, out_dir: Path) -> None:
    if not data:
        return
    all_ns = sorted({n for d in data.values() for n in d.keys()})
    for n in all_ns:
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        any_line = False
        for key in SERIES_ORDER:
            if key not in data or n not in data[key]:
                continue
            lats = _lat_arr(data[key][n])
            if lats.size == 0:
                continue
            # Downsample for CDF if huge — preserves shape, cuts plot time.
            if lats.size > 200_000:
                rng = np.random.default_rng(0)
                sample = rng.choice(lats, size=200_000, replace=False)
                lats = np.sort(sample)
            else:
                lats = np.sort(lats)
            ys = np.arange(1, lats.size + 1) / lats.size
            ax.plot(lats, ys,
                    color=SERIES_COLORS[key],
                    linestyle=SERIES_LINESTYLE[key],
                    label=SERIES_LABELS[key])
            any_line = True
        if not any_line:
            plt.close(fig)
            continue
        ax.set_xlabel("End-to-end latency (ms)")
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1.01)
        ax.set_title(f"N = {n}")
        ax.grid(True)
        ax.legend(loc="lower right", frameon=False)
        save_figure(fig, out_dir / f"latency_cdf_N{n}.pdf")
        plt.close(fig)


def _all_backbones(data: Dict[str, Dict[int, pd.DataFrame]]) -> List[str]:
    bbs: set = set()
    for cond in data.values():
        for df in cond.values():
            bbs.update(df["backbone"].unique().tolist())
    return sorted(bbs)


def plot_backbone_latency_vs_n(data, out_path: Path, stat: str) -> None:
    backbones = _all_backbones(data)
    if not backbones:
        return
    all_ns = sorted({n for d in data.values() for n in d.keys()})
    keys = [k for k in SERIES_ORDER if k in data]
    if not all_ns or not keys:
        return

    n_cols = min(2, len(backbones))
    n_rows = int(np.ceil(len(backbones) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(max(5.0, 3.1 * n_cols), 2.6 * n_rows),
                             sharex=True)
    axes_arr = np.atleast_1d(axes).flatten()

    for ax, backbone in zip(axes_arr, backbones):
        any_line = False
        for key in keys:
            ys = []
            for n in all_ns:
                df = data.get(key, {}).get(n)
                if df is None:
                    ys.append(np.nan)
                    continue
                mask = df["backbone"] == backbone
                lats = df.loc[mask, "end_to_end_latency(ms)"].to_numpy()
                ys.append(_stat_arr(lats, stat) if lats.size else np.nan)
            if np.all(np.isnan(ys)):
                continue
            ax.plot(all_ns, ys,
                    color=SERIES_COLORS[key],
                    linestyle=SERIES_LINESTYLE[key],
                    marker="o",
                    markersize=3.5,
                    label=SERIES_LABELS[key])
            any_line = True

        if not any_line:
            ax.set_visible(False)
            continue

        ax.set_title(backbone, pad=2)
        ax.grid(True, axis="y")
        ax.set_ylabel(f"{stat.upper()} latency (ms)")

    for ax in axes_arr[len(backbones):]:
        ax.set_visible(False)

    for ax in axes_arr[:len(backbones)]:
        ax.set_xlabel("Number of applications")

    handles, labels = [], []
    for ax in axes_arr[:len(backbones)]:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(handles)), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.5)
    else:
        fig.tight_layout(pad=0.5)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_success_rate_vs_n(run_status: Dict[str, Dict[int, Dict[str, float]]],
                           out_path: Path) -> None:
    if not run_status:
        return
    all_ns = sorted({n for d in run_status.values() for n in d.keys()})
    keys = [k for k in SERIES_ORDER if k in run_status]
    if not all_ns or not keys:
        return

    fig, ax = plt.subplots(figsize=(max(4.4, 0.65 * len(all_ns) + 1.4), 2.8))
    for key in keys:
        rates = [run_status[key].get(n, {}).get("success_rate", np.nan) for n in all_ns]
        if _series_is_all_zero_or_nan(rates):
            continue
        ax.plot(all_ns, rates,
                color=SERIES_COLORS[key],
                linestyle=SERIES_LINESTYLE[key],
                marker="o",
                markersize=4.0,
                markeredgewidth=0.0,
                solid_capstyle="round",
                solid_joinstyle="round",
                antialiased=True,
                label=SERIES_LABELS[key])
    ax.set_xticks(all_ns)
    ax.set_xlabel("Number of applications")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    ax.margins(x=0.04)
    ax.grid(True, axis="y")
    _plain_number_axis(ax, "y")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, out_path)
    plt.close(fig)


# ── Per-N detail plots ────────────────────────────────────────────

def _per_task_stats(df: pd.DataFrame, col: str, stat: str
                    ) -> Dict[str, float]:
    """{task -> stat(col)} via single groupby pass."""
    if df.empty:
        return {}
    g = df.groupby("task", observed=True)[col]
    if stat == "mean":
        s = g.mean()
    elif stat == "p50":
        s = g.quantile(0.50)
    elif stat == "p95":
        s = g.quantile(0.95)
    elif stat == "p99":
        s = g.quantile(0.99)
    else:
        raise ValueError(stat)
    return {str(k): float(v) for k, v in s.items()}


def _per_task_stat_plot(per_task: Dict[str, Dict[str, float]],
                        ylabel: str, out_path: Path) -> None:
    if not per_task:
        return
    tasks = sorted({t for d in per_task.values() for t in d.keys()})
    keys = [k for k in SERIES_ORDER if k in per_task]
    if not tasks or not keys:
        return

    fig, ax = plt.subplots(figsize=(max(4.5, 0.45 * len(tasks) + 1.5), 2.8))
    x = np.arange(len(tasks))
    width = 0.8 / len(keys)
    for i, k in enumerate(keys):
        vals = [per_task[k].get(t, 0.0) for t in tasks]
        ax.bar(x + (i - (len(keys) - 1) / 2) * width, vals, width,
               label=SERIES_LABELS[k], color=SERIES_COLORS[k])
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_summary_bars_for_n(data, n: int, out_path: Path) -> None:
    keys = [k for k in SERIES_ORDER if k in data and n in data[k]]
    if not keys:
        return
    metrics = ["mean", "p50", "p95", "p99"]
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    x = np.arange(len(metrics))
    width = 0.8 / len(keys)
    for i, k in enumerate(keys):
        lats = _lat_arr(data[k][n])
        vals = [_stat_arr(lats, m) for m in metrics]
        ax.bar(x + (i - (len(keys) - 1) / 2) * width, vals, width,
               label=SERIES_LABELS[k], color=SERIES_COLORS[k])
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylabel("End-to-end latency (ms)")
    ax.set_title(f"N = {n}")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, out_path)
    plt.close(fig)


def _nice_upper(vmax: float) -> float:
    """Round vmax up to a clean tick (fine ladder so 114 → 120, not 200)."""
    if not np.isfinite(vmax) or vmax <= 0:
        return 1.0
    mag = 10 ** np.floor(np.log10(vmax))
    # Fine 1.1× ladder: plenty of stops so the top tick stays close to vmax.
    ladder = (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10)
    for m in ladder:
        if m * mag >= vmax:
            return float(m * mag)
    return float(10 * mag)


def plot_latency_over_time_for_n(data, n: int, out_path: Path,
                                 bin_s: float = 1.0) -> None:
    """Mean response-time over time, one line per method, for a given N."""
    keys = [k for k in SERIES_ORDER if k in data and n in data[k]]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(1.8, 1.2))
    any_line = False
    x_max = 0.0
    y_max = 0.0
    # Shift each method's timeline to start at its own min req_time so the
    # axis starts at 0 instead of at the warmup cutoff.
    for k in keys:
        df = data[k][n]
        t = df["req_time"].to_numpy()
        lat = _lat_arr(df)
        if t.size == 0:
            continue
        t = t - float(t.min())
        t_end = float(t.max())
        n_bins = int(np.ceil(t_end / bin_s)) + 1
        if n_bins <= 0:
            continue
        idx = (t / bin_s).astype(np.int64)
        idx = np.clip(idx, 0, n_bins - 1)
        sums = np.bincount(idx, weights=lat, minlength=n_bins)
        counts = np.bincount(idx, minlength=n_bins).astype(float)
        means = np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=float),
                          where=counts > 0)
        centers = (np.arange(n_bins) + 0.5) * bin_s
        mask = np.isfinite(means)
        ax.plot(centers[mask], means[mask],
                color=SERIES_COLORS[k], linestyle=SERIES_LINESTYLE[k],
                linewidth=1.0, label=SERIES_LABELS[k])
        x_max = max(x_max, float(centers[mask].max()) if mask.any() else 0.0)
        if mask.any():
            y_max = max(y_max, float(means[mask].max()))
        any_line = True
    if not any_line:
        plt.close(fig)
        return
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean RT (ms)")
    ax.set_xlim(0, x_max if x_max > 0 else None)
    ax.set_ylim(0, _nice_upper(y_max))
    ax.grid(True, axis="both")
    ax.legend(frameon=False, ncol=min(3, len(keys)), handlelength=1.2,
              handletextpad=0.3, columnspacing=0.7,
              loc="lower center", bbox_to_anchor=(0.5, 1.02),
              borderaxespad=0.0)
    save_figure(fig, out_path)
    plt.close(fig)


def _bar_over_methods(data, n: int, stat: str, ylabel: str, out_path: Path) -> None:
    keys = [k for k in SERIES_ORDER if k in data and n in data[k]]
    if not keys:
        return
    labels = [SERIES_LABELS[k] for k in keys]
    vals = [_stat_arr(_lat_arr(data[k][n]), stat) for k in keys]
    colors = [SERIES_COLORS[k] for k in keys]
    fig, ax = plt.subplots(figsize=(1.8, 1.2))
    x = np.arange(len(keys))
    bars = ax.bar(x, vals, width=0.55, color=colors,
                  edgecolor="black", linewidth=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}",
                ha="center", va="bottom", fontsize=6.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    vmax = max(vals) if vals else 0.0
    # Leave a little headroom for the value labels above each bar.
    ax.set_ylim(0, _nice_upper(vmax * 1.10))
    ax.grid(True, axis="y")
    save_figure(fig, out_path)
    plt.close(fig)


def plot_per_n_details(data, out_root: Path) -> None:
    all_ns = sorted({n for d in data.values() for n in d.keys()})
    for n in all_ns:
        out_dir = out_root / f"N{n}"
        plot_summary_bars_for_n(data, n, out_dir / "summary_bars.pdf")
        plot_latency_over_time_for_n(data, n, out_dir / "response_time_vs_time.pdf")
        _bar_over_methods(data, n, "mean", "Mean RT (ms)",
                          out_dir / "mean_response_time_vs_method.pdf")
        _bar_over_methods(data, n, "p99", "p99 RT (ms)",
                          out_dir / "p99_response_time_vs_method.pdf")

        for stat, label in [("p50", "p50 latency (ms)"),
                            ("p95", "p95 latency (ms)"),
                            ("p99", "p99 latency (ms)")]:
            per_task_lat = {}
            for k in SERIES_ORDER:
                if k not in data or n not in data[k]:
                    continue
                per_task_lat[k] = _per_task_stats(
                    data[k][n], "end_to_end_latency(ms)", stat)
            _per_task_stat_plot(per_task_lat, label,
                                out_dir / f"per_task_{stat}_latency.pdf")

        per_task_exec = {}
        for k in SERIES_ORDER:
            if k not in data or n not in data[k]:
                continue
            per_task_exec[k] = _per_task_stats(
                data[k][n], "backend_exec_time(ms)", "mean")
        _per_task_stat_plot(per_task_exec,
                            "Mean backend exec time (ms)",
                            out_root / f"N{n}" / "per_task_service_time.pdf")


# ── Deployment diagram ─────────────────────────────────────────────

BACKBONE_ABBREV = {
    "momentsmall":    "MS",
    "momentbase":     "MB",
    "momentlarge":    "ML",
    "chronostiny":    "CT",
    "chronossmall":   "CS",
    "chronosbase":    "CB",
    "chronoslarge":   "CL",
    "dinosmall":      "DS",
    "dinobase":       "DB",
    "swinsmall":      "SS",
    "swinbase":       "SB",
    "papageissvri":   "PG",
    "papageisp":     "PP",
}

TASK_PASTELS = [
    "#ffffcc", "#d4f1c0", "#ffd6e0", "#d6eaff",
    "#ffe8cc", "#e8d5f5", "#ccf5f1", "#fce4d6",
    "#dce8d0", "#f5e6cc",
]
GPU_COLOR      = "#a8dde8"
BACKBONE_COLOR = "#fdd58a"
FMVISOR_COLOR  = "#cde5b8"


def _load_plan(cond_dir: Path) -> dict | None:
    p = cond_dir / "deployment_plan.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"[Plot] Failed to read {p}: {e}")
        return None


def _is_sharing_plan(plan: dict) -> bool:
    """True if any GPU hosts multiple backbones in one deployment (FMVisor)."""
    for site in plan.get("sites", []):
        for d in site.get("deployments", []):
            # FMVisor deployments carry multiple decoders under one backbone
            # AND there is exactly one deployment per (device_name, cuda).
            pass
    # Simpler: sharing iff each GPU has exactly one deployment and no tpc_partition.
    by_gpu: Dict[str, int] = {}
    any_tpc = False
    for site in plan.get("sites", []):
        for d in site.get("deployments", []):
            key = d.get("device_name", d["device"])
            by_gpu[key] = by_gpu.get(key, 0) + 1
            if isinstance(d.get("tpc_partition"), list) and d["tpc_partition"]:
                any_tpc = True
    return (not any_tpc) and all(v == 1 for v in by_gpu.values()) and bool(by_gpu)


def _plot_deployment_diagram(plan: dict, out_path: Path, title: str) -> None:
    # Collect deployments grouped by device_name.
    deployments = []
    for site in plan.get("sites", []):
        for d in site.get("deployments", []):
            deployments.append(d)
    if not deployments:
        return

    def _sort_key(d: dict):
        import re as _re
        dname = d.get("device_name", d.get("device", ""))
        m = _re.search(r"(\d+)$", dname)
        return (int(m.group(1)) if m else 999, dname, d.get("backbone", ""))

    deployments.sort(key=_sort_key)

    gpu_groups: Dict[str, List[dict]] = {}
    for d in deployments:
        gpu_groups.setdefault(d.get("device_name", d["device"]), []).append(d)

    gpu_labels = list(gpu_groups.keys())
    n_gpus     = len(gpu_labels)
    total_deps = sum(len(v) for v in gpu_groups.values())

    all_tasks = sorted({
        t for d in deployments
        for t in (
            [dec["task"] for dec in d.get("decoders", [])]
            if d.get("decoders")
            else list(d.get("tasks", {}).keys())
        )
    })
    task_pastel = {t: TASK_PASTELS[i % len(TASK_PASTELS)]
                   for i, t in enumerate(all_tasks)}
    tabbrev = {t: f"T{i+1}" for i, t in enumerate(all_tasks)}

    sharing = _is_sharing_plan(plan)

    TASK_H_IN = 0.20
    BB_H_IN   = 0.12
    FM_H_IN   = 0.10
    GPU_H_IN  = 0.10
    PAD_IN    = 0.025
    # Optional FMVisor band adds its own height + pad only when present.
    fm_extra = (FM_H_IN + PAD_IN) if sharing else 0.0
    fig_h = GPU_H_IN + PAD_IN + BB_H_IN + PAD_IN + fm_extra + TASK_H_IN + 0.08
    # Width fixed by #GPUs so every method renders at the exact same size.
    # We size for the worst case (2 deps/GPU) so no_sharing stays readable.
    fig_w = max(2.0, 0.85 * n_gpus + 0.4)

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def _frac(inches: float) -> float:
        return inches / fig_h

    GPU_H  = _frac(GPU_H_IN)
    BB_H   = _frac(BB_H_IN)
    FM_H   = _frac(FM_H_IN) if sharing else 0.0
    TASK_H = _frac(TASK_H_IN)
    PAD    = _frac(PAD_IN)
    BOTTOM = _frac(0.02)

    # Equal per-GPU width regardless of #deployments inside each GPU.
    GPU_GAP = 0.024
    SUB_GAP = 0.006
    gpu_col_w = (1.0 - GPU_GAP * max(n_gpus - 1, 0)) / max(n_gpus, 1)

    # If any deployment declares a tpc_partition, use TPC fractions to size
    # sub-columns within a GPU; otherwise split evenly.
    def _tpc_count(dev: dict) -> int | None:
        part = dev.get("tpc_partition")
        if isinstance(part, list) and part:
            return len(part)
        return None

    x_cursor = 0.0
    for gpu_label in gpu_labels:
        devs = gpu_groups[gpu_label]
        n_deps = len(devs)

        ax.add_patch(FancyBboxPatch(
            (x_cursor, BOTTOM), gpu_col_w, GPU_H, boxstyle="round,pad=0.006",
            facecolor=GPU_COLOR, edgecolor="#5aafc4", linewidth=0.6,
            transform=ax.transAxes, clip_on=False))
        ax.text(x_cursor + gpu_col_w / 2, BOTTOM + GPU_H / 2, gpu_label,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=6.0, fontweight="bold", color="black")

        tpc_counts = [_tpc_count(d) for d in devs]
        has_tpc = all(c is not None for c in tpc_counts) and sum(tpc_counts) > 0
        if has_tpc:
            total_tpc = float(sum(tpc_counts))
            avail = gpu_col_w - SUB_GAP * (n_deps - 1)
            sub_ws = [avail * (c / total_tpc) for c in tpc_counts]
        else:
            avail = gpu_col_w - SUB_GAP * (n_deps - 1)
            sub_ws = [avail / n_deps] * n_deps

        bb_y = BOTTOM + GPU_H + PAD
        fm_y = bb_y + BB_H + PAD
        task_y = fm_y + (FM_H + PAD if sharing else 0.0)

        # FMVisor band spans the whole GPU column, above the backbone(s).
        if sharing:
            ax.add_patch(FancyBboxPatch(
                (x_cursor, fm_y), gpu_col_w, FM_H, boxstyle="round,pad=0.004",
                facecolor=FMVISOR_COLOR, edgecolor="#6a9a52", linewidth=0.5,
                transform=ax.transAxes, clip_on=False))
            ax.text(x_cursor + gpu_col_w / 2, fm_y + FM_H / 2, "FMVisor",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=5.5, fontweight="bold", color="black")

        sub_x = x_cursor
        for dev, sw, tpc in zip(devs, sub_ws, tpc_counts):
            bb = dev.get("backbone", "")
            decoders = dev.get("decoders", [])
            ax.add_patch(FancyBboxPatch(
                (sub_x, bb_y), sw, BB_H, boxstyle="round,pad=0.004",
                facecolor=BACKBONE_COLOR, edgecolor="#c8950a", linewidth=0.5,
                transform=ax.transAxes, clip_on=False))
            # Auto-shrink backbone label so it never overflows the sub-column.
            bb_label = BACKBONE_ABBREV.get(bb, bb)
            avail_pt = sw * fig_w * 72.0 * 0.85  # 85% of box width in points
            label_pt = max(2, len(bb_label)) * 3.6  # ~3.6pt per char at 6pt
            bb_fs = min(6.0, max(4.0, 6.0 * avail_pt / label_pt))
            ax.text(sub_x + sw / 2, bb_y + BB_H / 2, bb_label,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=bb_fs, fontweight="bold", color="black")

            task_names = (
                [dec["task"] for dec in decoders]
                if decoders
                else list(dev.get("tasks", {}).keys())
            )
            n_t = max(len(task_names), 1)
            t_gap = 0.002
            t_w = (sw - t_gap * (n_t - 1)) / n_t
            for ti, task_name in enumerate(task_names):
                tx = sub_x + ti * (t_w + t_gap)
                fill = task_pastel.get(task_name, "#eeeeee")
                ax.add_patch(FancyBboxPatch(
                    (tx, task_y), t_w, TASK_H, boxstyle="round,pad=0.003",
                    facecolor=fill, edgecolor="#555555", linewidth=0.4,
                    linestyle="--",
                    transform=ax.transAxes, clip_on=False))
                ax.text(tx + t_w / 2, task_y + TASK_H / 2,
                        tabbrev.get(task_name, task_name),
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=5.0, fontweight="bold", color="black")

            sub_x += sw + SUB_GAP

        # Subtle dashed partition markers — kept short and semi-transparent
        # so device labels on the GPU bar remain readable.
        if has_tpc and n_deps > 1:
            total_tpc = float(sum(tpc_counts))
            cum = 0
            tick_h = GPU_H * 0.35
            for c in tpc_counts[:-1]:
                cum += c
                xline = x_cursor + gpu_col_w * (cum / total_tpc)
                # Top tick
                ax.plot([xline, xline],
                        [BOTTOM + GPU_H - tick_h, BOTTOM + GPU_H],
                        color="#1f4b5c", linewidth=0.5, alpha=0.55,
                        transform=ax.transAxes, clip_on=False)
                # Bottom tick
                ax.plot([xline, xline],
                        [BOTTOM, BOTTOM + tick_h],
                        color="#1f4b5c", linewidth=0.5, alpha=0.55,
                        transform=ax.transAxes, clip_on=False)

        x_cursor += gpu_col_w + GPU_GAP

    save_figure(fig, out_path)
    plt.close(fig)


def plot_deployments(exp_dir: Path, out_root: Path) -> None:
    """One deployment diagram per (N, condition) under perN/N{N}/."""
    for n_dir in sorted(exp_dir.iterdir()):
        m = N_DIR_RE.match(n_dir.name) if n_dir.is_dir() else None
        if not m:
            continue
        n = int(m.group(1))
        for cond_dir in sorted(n_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            plan = _load_plan(cond_dir)
            if plan is None:
                continue
            out_path = out_root / f"N{n}" / f"deployment_{cond_dir.name}.pdf"
            _plot_deployment_diagram(plan, out_path, "")


# ── Main ───────────────────────────────────────────────────────────

def plot_latency_vs_load(data: Dict[str, Dict[int, pd.DataFrame]],
                         out_path: Path,
                         window_s: float = 0.2) -> None:
    """Latency vs instantaneous device load, per condition, for one N.

    For each request we compute the RPS on its device in a small window
    around its arrival, then bin requests by that load and plot median and
    p99 end-to-end latency per bin. The knee of this curve shifts right
    under sharing (it tolerates higher instantaneous load) — that is the
    burst-absorption claim made visual.
    """
    conds = [c for c in SERIES_ORDER if c in data]
    if not conds:
        return
    ns_all = sorted({n for c in conds for n in data[c].keys()})
    if not ns_all:
        return
    # Pick the largest common N (best signal for bursts + colocation).
    common = set(data[conds[0]].keys())
    for c in conds[1:]:
        common &= set(data[c].keys())
    if not common:
        return
    n = max(common)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=False)
    for cond in conds:
        df = data[cond].get(n)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["t"] = df["req_time"].astype("float64")
        # Instantaneous per-device RPS: count requests on the same device
        # within ±window_s/2 of each request's arrival time.
        rps = np.zeros(len(df), dtype="float32")
        for dev, sub in df.groupby("device", observed=True):
            t = sub["t"].to_numpy()
            order = np.argsort(t)
            t_sorted = t[order]
            lo = np.searchsorted(t_sorted, t_sorted - window_s / 2, side="left")
            hi = np.searchsorted(t_sorted, t_sorted + window_s / 2, side="right")
            cnt = (hi - lo) / window_s
            out = np.empty_like(cnt)
            out[order] = cnt
            rps[sub.index.to_numpy()] = out
        df["inst_rps"] = rps
        lat = df["end_to_end_latency(ms)"].to_numpy()

        # Bin by load quantile for stable x support across methods.
        q_edges = np.quantile(df["inst_rps"], np.linspace(0, 1, 21))
        q_edges = np.unique(q_edges)
        if len(q_edges) < 3:
            continue
        idx = np.digitize(df["inst_rps"], q_edges[1:-1], right=True)
        bins = pd.DataFrame({"bin": idx, "rps": df["inst_rps"], "lat": lat})
        agg = bins.groupby("bin").agg(
            x=("rps", "mean"),
            p50=("lat", "median"),
            p99=("lat", lambda s: np.quantile(s, 0.99)),
            n=("lat", "size"),
        ).reset_index()
        agg = agg[agg["n"] >= 30]
        if agg.empty:
            continue

        color = SERIES_COLORS.get(cond, "#888888")
        label = SERIES_LABELS.get(cond, cond)
        axes[0].plot(agg["x"], agg["p50"], marker="o", ms=3.5, color=color,
                     linestyle=SERIES_LINESTYLE.get(cond, "-"), label=label)
        axes[1].plot(agg["x"], agg["p99"], marker="o", ms=3.5, color=color,
                     linestyle=SERIES_LINESTYLE.get(cond, "-"), label=label)

    for ax, stat in zip(axes, ["median", "p99"]):
        ax.set_xlabel(f"Instantaneous per-device load (req/s, {int(window_s*1000)} ms window)")
        ax.set_ylabel(f"End-to-end latency ({stat}, ms)")
        ax.grid(alpha=0.3)
    axes[0].set_title(f"Median latency vs load (N={n})")
    axes[1].set_title(f"p99 latency vs load (N={n})")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_backbone_exec_bars(data: Dict[str, Dict[int, pd.DataFrame]],
                            out_path: Path) -> None:
    """Grouped bar chart: mean backbone exec time per batch, per backbone per N.

    X-axis: N values.  One subplot per backbone.  One bar-group per N,
    one bar per condition.  Matches the style of _bars_vs_n / latency_bars_mean.
    """
    backbone_order  = ["dinosmall", "momentbase", "swinsmall", "papageissvri"]
    backbone_labels = {
        "dinosmall":    "DINOv2-S",
        "momentbase":   "MOMENT-B",
        "swinsmall":    "Swin-S",
        "papageissvri": "Papagei",
    }

    keys   = [k for k in SERIES_ORDER if k in data]
    all_ns = sorted({n for k in keys for n in data[k]})
    if not keys or not all_ns:
        return

    # Compute per-batch mean proc_time for every (cond, N, backbone)
    # proc_time is identical for every request in the same batch, so take
    # the first value per (device, device_start_time) group.
    vals: Dict[str, Dict[int, Dict[str, float]]] = {}
    for cond in keys:
        vals[cond] = {}
        for n in all_ns:
            df = data[cond].get(n)
            if df is None or df.empty:
                continue
            vals[cond][n] = {}
            for bb, grp in df.groupby("backbone"):
                vals[cond][n][str(bb)] = float(
                    grp.groupby(["device", "device_start_time"])["proc_time(ms)"]
                       .first().mean()
                )

    backbones = [b for b in backbone_order
                 if any(b in vals[c].get(n, {}) for c in keys for n in all_ns)]
    if not backbones:
        return

    n_bb   = len(backbones)
    x      = np.arange(len(all_ns), dtype=float)
    width  = 0.8 / len(keys)

    fig, axes = plt.subplots(1, n_bb, figsize=(1.8 * n_bb, 1.2), sharey=False)
    if n_bb == 1:
        axes = [axes]

    for ax, bb in zip(axes, backbones):
        vmax = 0.0
        for i, cond in enumerate(keys):
            ys = [vals[cond].get(n, {}).get(bb, 0.0) for n in all_ns]
            if _series_is_all_zero_or_nan(ys):
                continue
            bars = ax.bar(
                x + (i - (len(keys) - 1) / 2) * width, ys, width,
                color=SERIES_COLORS[cond], edgecolor="black", linewidth=0.4,
                label=SERIES_LABELS[cond],
            )
            for b, v in zip(bars, ys):
                if v > 0 and np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                            f"{v:.0f}", ha="center", va="bottom", fontsize=5.0)
            vmax = max(vmax, max((v for v in ys if np.isfinite(v)), default=0.0))

        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in all_ns])
        ax.set_xlabel("N (apps)")
        ax.set_title(backbone_labels.get(bb, bb), fontsize=7, pad=2)
        ax.set_ylim(0, _nice_upper(vmax * 1.12))
        ax.grid(True, axis="y")
        if ax is axes[0]:
            ax.set_ylabel("Backbone exec (ms)")

    # Single shared legend above all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False,
               ncol=len(keys), handlelength=1.0, handletextpad=0.3,
               columnspacing=0.7, loc="lower center",
               bbox_to_anchor=(0.5, 1.02), borderaxespad=0.0,
               fontsize=6)
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_backbone_exec_bars_per_n(data: Dict[str, Dict[int, pd.DataFrame]],
                                   out_dir: Path) -> None:
    """Per-N backbone exec bar chart: x-axis = backbones, grouped bars per condition.

    Saves backbone_exec_bars_N{n}.pdf for every N that has data.
    """
    backbone_order  = ["dinosmall", "momentbase", "swinsmall", "papageissvri"]
    backbone_labels = {
        "dinosmall":    "DINOv2-S",
        "momentbase":   "MOMENT-B",
        "swinsmall":    "Swin-S",
        "papageissvri": "Papagei",
    }

    keys   = [k for k in SERIES_ORDER if k in data]
    all_ns = sorted({n for k in keys for n in data[k]})
    if not keys or not all_ns:
        return

    # Pre-compute per-batch mean proc_time for every (cond, N, backbone)
    vals: Dict[str, Dict[int, Dict[str, float]]] = {}
    for cond in keys:
        vals[cond] = {}
        for n in all_ns:
            df = data[cond].get(n)
            if df is None or df.empty:
                continue
            vals[cond][n] = {}
            for bb, grp in df.groupby("backbone"):
                vals[cond][n][str(bb)] = float(
                    grp.groupby(["device", "device_start_time"])["proc_time(ms)"]
                       .first().mean()
                )

    for n in all_ns:
        backbones = [b for b in backbone_order
                     if any(b in vals[c].get(n, {}) for c in keys)]
        if not backbones:
            continue

        n_bb  = len(backbones)
        x     = np.arange(n_bb, dtype=float)
        width = 0.8 / len(keys)

        fig, ax = plt.subplots(figsize=(1.8, 1.2))
        vmax = 0.0
        for i, cond in enumerate(keys):
            ys = [vals[cond].get(n, {}).get(bb, 0.0) for bb in backbones]
            if _series_is_all_zero_or_nan(ys):
                continue
            bars = ax.bar(
                x + (i - (len(keys) - 1) / 2) * width, ys, width,
                color=SERIES_COLORS[cond], edgecolor="black", linewidth=0.4,
                label=SERIES_LABELS[cond],
            )
            for b, v in zip(bars, ys):
                if v > 0 and np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                            f"{v:.0f}", ha="center", va="bottom", fontsize=5.0)
            vmax = max(vmax, max((v for v in ys if np.isfinite(v)), default=0.0))

        ax.set_xticks(x)
        ax.set_xticklabels([backbone_labels.get(bb, bb) for bb in backbones],
                           rotation=30, ha="right", fontsize=5.5)
        ax.set_ylabel("Backbone exec (ms)")
        ax.set_ylim(0, _nice_upper(vmax * 1.12))
        ax.grid(True, axis="y")

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False,
                   ncol=len(keys), handlelength=1.0, handletextpad=0.3,
                   columnspacing=0.7, loc="lower center",
                   bbox_to_anchor=(0.5, 1.02), borderaxespad=0.0,
                   fontsize=6)
        fig.tight_layout()
        save_figure(fig, out_dir / f"backbone_exec_bars_N{n}.pdf")
        plt.close(fig)


def plot_backbone_decoder_breakdown(data: Dict[str, Dict[int, pd.DataFrame]],
                                     out_path: Path) -> None:
    """Stacked bars of mean backbone vs decoder exec time, per (N, condition).

    Also annotates each bar with the request-weighted backbone batch size.
    Shows that sharing amortizes the backbone forward pass (proc_time) across
    tasks, while decoder_time stays roughly flat — i.e. the gain is kernel
    fusion at the shared trunk, not at the per-task heads.
    """
    conds = [c for c in SERIES_ORDER if c in data]
    if not conds:
        return
    ns = sorted({n for c in conds for n in data[c].keys()})
    if not ns:
        return

    rows = []
    for cond in conds:
        for n in ns:
            df = data[cond].get(n)
            if df is None or df.empty:
                continue
            g = df.groupby(["device", "device_start_time"]).size().rename("bs")
            d = df.merge(g, on=["device", "device_start_time"])
            rows.append({
                "cond": cond,
                "N": n,
                "backbone_ms": float(df["proc_time(ms)"].mean()),
                "decoder_ms": float(df["decoder_time(ms)"].mean()),
                "bs_rw": float((d["bs"] * d["bs"]).sum() / d["bs"].sum()),
            })
    if not rows:
        return
    summary = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(max(6.5, 1.2 * len(ns) + 2), 4.2))
    width = 0.8 / len(conds)
    x = np.arange(len(ns))
    for i, cond in enumerate(conds):
        sub = summary[summary["cond"] == cond].set_index("N").reindex(ns)
        offset = (i - (len(conds) - 1) / 2) * width
        color = SERIES_COLORS.get(cond, "#888888")
        # darker = backbone, lighter = decoder
        ax.bar(x + offset, sub["backbone_ms"], width,
               color=color, edgecolor="black", linewidth=0.4,
               label=f"{SERIES_LABELS.get(cond, cond)} — backbone")
        ax.bar(x + offset, sub["decoder_ms"], width,
               bottom=sub["backbone_ms"],
               color=color, alpha=0.35, edgecolor="black", linewidth=0.4,
               label=f"{SERIES_LABELS.get(cond, cond)} — decoder")
        for xi, n in enumerate(ns):
            if pd.isna(sub.loc[n, "backbone_ms"]):
                continue
            top = sub.loc[n, "backbone_ms"] + sub.loc[n, "decoder_ms"]
            ax.text(x[xi] + offset, top + 0.6,
                    f"b={sub.loc[n, 'bs_rw']:.1f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in ns])
    ax.set_ylabel("Mean per-request exec time (ms)")
    ax.set_title("Backbone vs decoder exec time  (b = request-weighted batch size)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, ncol=len(conds), loc="upper left")
    fig.tight_layout()
    save_figure(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=str,
                        default=str(SERVING_DIR / "experiments/cluster_sharing_benefit/results_alibaba"))
    parser.add_argument("--warmup-secs", type=float, default=10.0)
    parser.add_argument("--ns", type=str, default="8,16,32,64,128,20,40,80,160",
                        help="Comma-separated list of N values to include "
                             "(e.g. --ns 8,16,24). Default: 8,16,32,64,128.")
    args = parser.parse_args()

    ns_filter: set[int] | None = None
    if args.ns:
        ns_filter = {int(x) for x in args.ns.split(",") if x.strip()}

    exp_dir = Path(args.exp_dir).resolve()
    if not exp_dir.exists():
        print(f"[Plot] exp-dir not found: {exp_dir}")
        return

    apply_paper_style()
    data = _scan(exp_dir, args.warmup_secs)
    run_status = _scan_run_status(exp_dir)
    if not data:
        print("[Plot] no data found.")
        return

    if ns_filter is not None:
        data = {k: {n: df for n, df in v.items() if n in ns_filter}
                for k, v in data.items()}
        data = {k: v for k, v in data.items() if v}
        run_status = {k: {n: s for n, s in v.items() if n in ns_filter}
                      for k, v in run_status.items()}
        run_status = {k: v for k, v in run_status.items() if v}
        if not data:
            print(f"[Plot] no data after --ns filter {sorted(ns_filter)}")
            return
        print(f"[Plot] filtering to N in {sorted(ns_filter)}")

    out_dir = exp_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_latency_vs_n(data, out_dir)
    plot_latency_bars_vs_n(data, out_dir)
    plot_memory_bars_vs_n(exp_dir, out_dir / "memory_bars_vs_napps.pdf",
                          ns_filter=ns_filter)
    plot_runtime_memory_bars_vs_n(exp_dir,
                                  out_dir / "runtime_memory_bars_vs_napps.pdf",
                                  ns_filter=ns_filter)
    plot_throughput_vs_n(data, out_dir / "throughput_vs_napps.pdf",
                         warmup_secs=args.warmup_secs)
    plot_backbone_latency_vs_n(data, out_dir / "backbone_latency_vs_n_mean.pdf", "mean")
    plot_backbone_latency_vs_n(data, out_dir / "backbone_latency_vs_n_p95.pdf", "p95")
    plot_success_rate_vs_n(run_status, out_dir / "success_rate_vs_napps.pdf")
    plot_latency_cdfs(data, out_dir)
    plot_backbone_decoder_breakdown(data, out_dir / "backbone_decoder_breakdown.pdf")
    plot_backbone_exec_bars(data, out_dir / "backbone_exec_bars.pdf")
    plot_backbone_exec_bars_per_n(data, out_dir)
    plot_latency_vs_load(data, out_dir / "latency_vs_load.pdf")
    plot_per_n_details(data, out_dir / "perN")
    plot_deployments(exp_dir, out_dir / "perN")
    print(f"[Plot] Done. Output in {out_dir}")


if __name__ == "__main__":
    main()
