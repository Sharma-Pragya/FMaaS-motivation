"""
Cross-scheduler comparison plots for FMaaS paper.

Reads results from multiple schedulers and request rates, then generates
side-by-side comparison plots for:
  1. SLO violation rate (% requests exceeding latency bound)
  2. Tail latency (p50, p95, p99) per scheduler
  3. Backbone memory footprint (total backbone MB loaded)
  4. Number of backbone replicas / deployments
  5. Response time breakdown (proc + swap + other)
  6. Throughput (achieved req/s)
  7. Accuracy / MAE metrics

Usage:
    python compare_schedulers.py --exp-dir experiments/baselines
    python compare_schedulers.py --exp-dir experiments/baselines --schedulers fmaas clipper m4
    python compare_schedulers.py --exp-dir experiments/baselines --req-rates 10 50 100 200
    python compare_schedulers.py --exp-dir experiments/baselines --format pdf
"""

import os
from sched import scheduler
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

# ── Styling ──────────────────────────────────────────────────────────────────

SCHEDULER_COLORS = {
    "fmaas":       "#2ca02c",   # green
    "fmaas_share": "#17becf",   # teal
    "clipper-ht":     "#1f77b4",   # blue
    "clipper-ha":     "#9467bd",   # purple
    "m4-ht":          "#ff7f0e",   # orange
    "m4-ha":          "#d62728",   # red
}

SCHEDULER_LABELS = {
    "fmaas":       "FMaaS-HA",
    "fmaas_share": "FMaaS-HS",
    "clipper-ht":     "Clipper-HT",
    "clipper-ha":     "Clipper-HA",
    "m4-ht":          "M4-HT",
    "m4-ha":          "M4-HA",
}

SCHEDULER_MARKERS = {
    "fmaas":       "D",
    "fmaas_share": "d",
    "clipper-ht":     "o",
    "clipper-ha":     "v",
    "m4-ht":          "s",
    "m4-ha":          "^",

}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_latency_csv(exp_dir, scheduler, req_rate):
    """Load request_latency_results.csv for a given scheduler and req_rate."""
    path = os.path.join(exp_dir, scheduler, str(req_rate), "request_latency_results.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_deployment_plan(exp_dir, scheduler, req_rate):
    """Load deployment_plan.json for a given scheduler and req_rate."""
    path = os.path.join(exp_dir, scheduler, str(req_rate), "deployment_plan.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_task_slos(config_path=None):
    """Load per-task latency SLOs from user_config.
    
    Returns dict: {task_name: latency_bound_ms}
    """
    if config_path is None:
        # Try to import from the standard location
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            from exp4.user_config import tasks
            return {name: spec.get('latency', float('inf')) for name, spec in tasks.items()}
        except ImportError:
            return {}
    else:
        # Load from provided path
        import importlib.util
        spec = importlib.util.spec_from_file_location("user_config", config_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return {name: s.get('latency', float('inf')) for name, s in mod.tasks.items()}


def discover_req_rates(exp_dir, schedulers):
    """Auto-discover available request rates from directory structure."""
    rates = set()
    for sched in schedulers:
        sched_dir = os.path.join(exp_dir, sched)
        if os.path.isdir(sched_dir):
            for d in os.listdir(sched_dir):
                if d.isdigit():
                    rates.add(int(d))
    return sorted(rates)


def discover_schedulers(exp_dir):
    """Auto-discover schedulers from directory structure."""
    known = list(SCHEDULER_LABELS.keys())
    found = []
    for d in os.listdir(exp_dir):
        if d in known and os.path.isdir(os.path.join(exp_dir, d)):
            found.append(d)
    return found if found else known


# ── Profiler data (for memory / metric computation) ──────────────────────────

def load_profiler_data():
    """Try to load profiler data for backbone memory and metric lookups."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from planner.parser.profiler import components, pipelines, metric
        return components, pipelines, metric
    except ImportError:
        return None, None, None


# ── Plot functions ───────────────────────────────────────────────────────────

def plot_slo_violation_rate(exp_dir, schedulers, req_rates, task_slos, output_dir, fmt):
    """Plot 1: SLO violation rate (%) per scheduler across request rates."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for sched in schedulers:
        violation_rates = []
        valid_rates = []
        for rr in req_rates:
            df = load_latency_csv(exp_dir, sched, rr)
            if df is None or df.empty:
                continue
            total = len(df)
            violated = 0
            for task, slo in task_slos.items():
                task_df = df[df["task"] == task]
                violated += (task_df["end_to_end_latency(ms)"] > slo).sum()
            violation_rates.append(100.0 * violated / total)
            valid_rates.append(rr)

        if valid_rates:
            ax.plot(valid_rates, violation_rates,
                    marker=SCHEDULER_MARKERS.get(sched, "o"),
                    color=SCHEDULER_COLORS.get(sched, "gray"),
                    label=SCHEDULER_LABELS.get(sched, sched),
                    linewidth=2, markersize=8)

    ax.set_xlabel("Request Rate (req/s)", fontsize=12, fontweight='bold')
    ax.set_ylabel("SLO Violation Rate (%)", fontsize=12, fontweight='bold')
    ax.set_title("SLO Violation Rate vs. Request Rate", fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(req_rates)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"slo_violation_rate.{fmt}"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: slo_violation_rate.{fmt}")


def plot_tail_latency(exp_dir, schedulers, req_rates, output_dir, fmt):
    """Plot 2: Tail latency (p50, p95, p99) per scheduler."""
    percentiles = [50, 95, 99]
    fig, axes = plt.subplots(1, len(percentiles), figsize=(6 * len(percentiles), 6), sharey=False)

    for pi, pct in enumerate(percentiles):
        ax = axes[pi]
        for sched in schedulers:
            vals = []
            valid_rates = []
            for rr in req_rates:
                df = load_latency_csv(exp_dir, sched, rr)
                if df is None or df.empty:
                    continue
                vals.append(np.percentile(df["end_to_end_latency(ms)"], pct))
                valid_rates.append(rr)

            if valid_rates:
                ax.plot(valid_rates, vals,
                        marker=SCHEDULER_MARKERS.get(sched, "o"),
                        color=SCHEDULER_COLORS.get(sched, "gray"),
                        label=SCHEDULER_LABELS.get(sched, sched),
                        linewidth=2, markersize=8)

        ax.set_xlabel("Request Rate (req/s)", fontsize=11)
        ax.set_ylabel(f"p{pct} Latency (ms)", fontsize=11)
        ax.set_title(f"p{pct} Latency", fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(req_rates)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"tail_latency.{fmt}"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: tail_latency.{fmt}")


def plot_backbone_memory(exp_dir, schedulers, req_rates, output_dir, fmt):
    """Plot 3: Total backbone memory footprint across cluster."""
    components, pipelines, _ = load_profiler_data()
    if components is None:
        print("  Skipped backbone_memory (profiler data not available)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for sched in schedulers:
        mem_vals = []
        valid_rates = []
        for rr in req_rates:
            plan = load_deployment_plan(exp_dir, sched, rr)
            if plan is None:
                continue
            total_mem = 0.0
            for site in plan.get("sites", []):
                for dep in site.get("deployments", []):
                    bb = dep.get("backbone", "")
                    if bb in components:
                        total_mem += components[bb]["mem"]
            mem_vals.append(total_mem)
            valid_rates.append(rr)

        if valid_rates:
            ax.plot(valid_rates, mem_vals,
                    marker=SCHEDULER_MARKERS.get(sched, "o"),
                    color=SCHEDULER_COLORS.get(sched, "gray"),
                    label=SCHEDULER_LABELS.get(sched, sched),
                    linewidth=2, markersize=8)

    ax.set_xlabel("Request Rate (req/s)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Total Backbone Memory (MB)", fontsize=12, fontweight='bold')
    ax.set_title("Backbone Memory Footprint vs. Request Rate", fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(req_rates)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"backbone_memory.{fmt}"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: backbone_memory.{fmt}")


def plot_num_deployments(exp_dir, schedulers, req_rates, output_dir, fmt):
    """Plot 4: Number of deployments (backbone replicas) per scheduler."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for sched in schedulers:
        dep_counts = []
        valid_rates = []
        for rr in req_rates:
            plan = load_deployment_plan(exp_dir, sched, rr)
            if plan is None:
                continue
            count = sum(len(site.get("deployments", [])) for site in plan.get("sites", []))
            dep_counts.append(count)
            valid_rates.append(rr)

        if valid_rates:
            ax.plot(valid_rates, dep_counts,
                    marker=SCHEDULER_MARKERS.get(sched, "o"),
                    color=SCHEDULER_COLORS.get(sched, "gray"),
                    label=SCHEDULER_LABELS.get(sched, sched),
                    linewidth=2, markersize=8)

    ax.set_xlabel("Request Rate (req/s)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Deployments", fontsize=12, fontweight='bold')
    ax.set_title("Deployments vs. Request Rate", fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(req_rates)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"num_deployments.{fmt}"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: num_deployments.{fmt}")


def plot_throughput(exp_dir, schedulers, req_rates, output_dir, fmt):
    """Plot 5: Achieved throughput per scheduler."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for sched in schedulers:
        throughputs = []
        valid_rates = []
        for rr in req_rates:
            df = load_latency_csv(exp_dir, sched, rr)
            if df is None or df.empty:
                continue
            # Throughput = total requests / trace duration
            t_min = df["req_time"].min()
            t_max = df["req_time"].max()
            duration = t_max - t_min if t_max > t_min else 1.0
            throughputs.append(len(df) / duration)
            valid_rates.append(rr)

        if valid_rates:
            ax.plot(valid_rates, throughputs,
                    marker=SCHEDULER_MARKERS.get(sched, "o"),
                    color=SCHEDULER_COLORS.get(sched, "gray"),
                    label=SCHEDULER_LABELS.get(sched, sched),
                    linewidth=2, markersize=8)

    # Ideal line
    if req_rates:
        ax.plot(req_rates, req_rates, linestyle='--', color='gray', alpha=0.5, label='Ideal')

    ax.set_xlabel("Request Rate (req/s)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Achieved Throughput (req/s)", fontsize=12, fontweight='bold')
    ax.set_title("Throughput vs. Request Rate", fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(req_rates)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"throughput.{fmt}"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: throughput.{fmt}")


def plot_response_time_breakdown(exp_dir, schedulers, req_rates, output_dir, fmt):
    """Plot 6: Response time breakdown (proc + swap + other) with req_rate on x-axis, grouped by scheduler."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(req_rates))
    width = 0.8 / len(schedulers)
    
    # Define hatches for time types
    hatches = {'proc': '//', 'swap': 'xx', 'other': '..'}
    
    for sched_idx, sched in enumerate(schedulers):
        proc_vals, swap_vals, other_vals = [], [], []
        
        for rr in req_rates:
            df = load_latency_csv(exp_dir, sched, rr)
            if df is None or df.empty:
                proc_vals.append(0)
                swap_vals.append(0)
                other_vals.append(0)
                continue
            
            p = df["proc_time(ms)"].mean()
            s = df["swap_time(ms)"].mean()
            e = df["end_to_end_latency(ms)"].mean()
            proc_vals.append(p)
            swap_vals.append(s)
            other_vals.append(max(e - p - s, 0.0))
        
        offset = (sched_idx - len(schedulers) / 2 + 0.5) * width
        
        # Plot proc bars
        proc_bars = ax.bar(x + offset, proc_vals, width,
                          color=SCHEDULER_COLORS.get(sched, "gray"), 
                          hatch=hatches['proc'], alpha=0.8, edgecolor="black", linewidth=0.5)
        
        # Plot swap bars
        swap_bars = ax.bar(x + offset, swap_vals, width, bottom=proc_vals,
                          color=SCHEDULER_COLORS.get(sched, "gray"), 
                          hatch=hatches['swap'], alpha=0.8, edgecolor="black", linewidth=0.5)
        
        # Plot other bars
        other_bars = ax.bar(x + offset, other_vals, width,
                           bottom=np.array(proc_vals) + np.array(swap_vals),
                           color=SCHEDULER_COLORS.get(sched, "gray"), 
                           hatch=hatches['other'], alpha=0.8, edgecolor="black", linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(rr) for rr in req_rates])
    ax.set_xlabel("Request Rate (req/s)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Mean Latency (ms)", fontsize=12, fontweight='bold')
    ax.set_title("Response Time Breakdown across Request Rates", fontsize=14, fontweight='bold')
    
    # Create two separate legends
    from matplotlib.patches import Patch
    
    # Legend 1: Schedulers (colors)
    scheduler_patches = [Patch(facecolor=SCHEDULER_COLORS.get(sched, "gray"), 
                               edgecolor="black", linewidth=0.5,
                               label=SCHEDULER_LABELS.get(sched, sched))
                        for sched in schedulers]
    legend1 = ax.legend(handles=scheduler_patches, loc='upper left', fontsize=10, title='Scheduler')
    ax.add_artist(legend1)
    
    # Legend 2: Time types (hatches)
    time_patches = [Patch(facecolor='gray', hatch=hatches['proc'], 
                          edgecolor="black", linewidth=0.5, label='Processing'),
                    Patch(facecolor='gray', hatch=hatches['swap'], 
                          edgecolor="black", linewidth=0.5, label='Swap'),
                    Patch(facecolor='gray', hatch=hatches['other'], 
                          edgecolor="black", linewidth=0.5, label='Other')]
    legend2 = ax.legend(handles=time_patches, loc='upper right', fontsize=10, title='Time Type')
    
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"response_breakdown.{fmt}"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: response_breakdown.{fmt}")


def plot_latency_boxplot(exp_dir, schedulers, req_rates, output_dir, fmt):
    """Plot 7: Latency box plots across request rates with all schedulers together."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(req_rates))
    width = 0.8 / len(schedulers)
    
    all_data = []
    all_labels = []
    all_colors = []
    
    for sched_idx, sched in enumerate(schedulers):
        positions = []
        data_list = []
        
        for rr_idx, rr in enumerate(req_rates):
            df = load_latency_csv(exp_dir, sched, rr)
            if df is None or df.empty:
                continue
            
            offset = (sched_idx - len(schedulers) / 2 + 0.5) * width
            pos = rr_idx + offset
            positions.append(pos)
            data_list.append(df["end_to_end_latency(ms)"].values)
        
        if data_list:
            bp = ax.boxplot(data_list, positions=positions, widths=width * 0.8,
                           patch_artist=True, showmeans=True, showfliers=False,
                           meanprops=dict(marker="D", markerfacecolor="black", 
                                        markeredgecolor="black", markersize=4),
                           medianprops=dict(color="black", linewidth=1.5),
                           boxprops=dict(facecolor=SCHEDULER_COLORS.get(sched, "gray"), 
                                        alpha=0.7, edgecolor="black"),
                           whiskerprops=dict(color="black", linewidth=0.8),
                           capprops=dict(color="black", linewidth=0.8))
            
            # Create custom legend entry
            for patch in bp["boxes"]:
                patch.set_label(SCHEDULER_LABELS.get(sched, sched))
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(rr) for rr in req_rates])
    ax.set_xlabel("Request Rate (req/s)", fontsize=12, fontweight='bold')
    ax.set_ylabel("End-to-End Latency (ms)", fontsize=12, fontweight='bold')
    ax.set_title("Latency Distribution across Request Rates", fontsize=14, fontweight='bold')
    
    # Create manual legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=SCHEDULER_COLORS.get(sched, "gray"), 
                            edgecolor="black", alpha=0.7,
                            label=SCHEDULER_LABELS.get(sched, sched))
                       for sched in schedulers]
    ax.legend(handles=legend_elements, loc='best', fontsize=10)
    
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"latency_boxplot.{fmt}"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: latency_boxplot.{fmt}")


def plot_per_task_slo(exp_dir, schedulers, req_rates, task_slos, output_dir, fmt):
    """Plot 8: Per-task SLO violation rate for each scheduler (grouped bar)."""
    for rr in req_rates:
        dfs = {}
        for sched in schedulers:
            df = load_latency_csv(exp_dir, sched, rr)
            if df is not None and not df.empty:
                dfs[sched] = df

        if not dfs:
            continue

        tasks = sorted(task_slos.keys())
        sched_names = [s for s in schedulers if s in dfs]
        n_sched = len(sched_names)
        n_tasks = len(tasks)

        if n_sched == 0 or n_tasks == 0:
            continue

        fig, ax = plt.subplots(figsize=(max(10, n_tasks * 2), 6))
        x = np.arange(n_tasks)
        width = 0.8 / n_sched

        for i, sched in enumerate(sched_names):
            df = dfs[sched]
            viol_rates = []
            for task in tasks:
                slo = task_slos.get(task, float('inf'))
                task_df = df[df["task"] == task]
                if len(task_df) == 0:
                    viol_rates.append(0.0)
                else:
                    viol_rates.append(100.0 * (task_df["end_to_end_latency(ms)"] > slo).sum() / len(task_df))

            ax.bar(x + i * width, viol_rates, width,
                   color=SCHEDULER_COLORS.get(sched, "gray"),
                   label=SCHEDULER_LABELS.get(sched, sched),
                   edgecolor="black", linewidth=0.5)

        ax.set_xticks(x + width * (n_sched - 1) / 2)
        ax.set_xticklabels(tasks, rotation=30, ha='right')
        ax.set_ylabel("SLO Violation Rate (%)", fontsize=12, fontweight='bold')
        ax.set_title(f"Per-Task SLO Violations @ {rr} req/s", fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"per_task_slo_{rr}.{fmt}"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: per_task_slo_{rr}.{fmt}")


def print_summary_table(exp_dir, schedulers, req_rates, task_slos):
    """Print a text summary table of key metrics."""
    print("\n" + "=" * 100)
    print(f"{'Scheduler':<18} {'Rate':>6} {'#Reqs':>7} {'#Deploy':>8} "
          f"{'p50(ms)':>9} {'p95(ms)':>9} {'p99(ms)':>9} {'SLO Viol%':>10} {'Thpt(r/s)':>10}")
    print("-" * 100)

    for sched in schedulers:
        for rr in req_rates:
            df = load_latency_csv(exp_dir, sched, rr)
            plan = load_deployment_plan(exp_dir, sched, rr)

            if df is None:
                continue

            n_reqs = len(df)
            n_dep = sum(len(s.get("deployments", [])) for s in plan.get("sites", [])) if plan else "?"

            lat = df["end_to_end_latency(ms)"]
            p50 = np.percentile(lat, 50)
            p95 = np.percentile(lat, 95)
            p99 = np.percentile(lat, 99)

            total_viol = 0
            for task, slo in task_slos.items():
                task_df = df[df["task"] == task]
                total_viol += (task_df["end_to_end_latency(ms)"] > slo).sum()
            viol_pct = 100.0 * total_viol / n_reqs if n_reqs > 0 else 0.0

            t_min, t_max = df["req_time"].min(), df["req_time"].max()
            dur = t_max - t_min if t_max > t_min else 1.0
            thpt = n_reqs / dur

            label = SCHEDULER_LABELS.get(sched, sched)
            print(f"{label:<18} {rr:>6} {n_reqs:>7} {n_dep:>8} "
                  f"{p50:>9.2f} {p95:>9.2f} {p99:>9.2f} {viol_pct:>9.2f}% {thpt:>10.2f}")

    print("=" * 100)

def plot_metric_comparison(exp_dir, schedulers, req_rates, output_dir, fmt):
    """Plot: Accuracy and MAE comparison across schedulers and request rates (separate plots)."""
    components, pipelines, metric = load_profiler_data()
    if components is None or pipelines is None or metric is None:
        print("  Skipped metric comparison (profiler data not available)")
        return
    
    # Organize data as {scheduler: {'acc': [...], 'mae': [...]}}
    sched_data = {}
    
    for sched in schedulers:
        sched_data[sched] = {'acc': [], 'mae': []}
        for rr in req_rates:
            deploymentplan = load_deployment_plan(exp_dir, sched, rr)
            if deploymentplan is None:
                sched_data[sched]['acc'].append(0)
                sched_data[sched]['mae'].append(0)
                continue
            
            sites = deploymentplan.get('sites', [])
            if not sites or 'deployments' not in sites[0]:
                sched_data[sched]['acc'].append(0)
                sched_data[sched]['mae'].append(0)
                continue
            
            deployments = sites[0]['deployments']
            acc, acc_count = 0, 0
            mae, mae_count = 0, 0
            
            for d in deployments:
                backbone = d['backbone']
                if backbone not in components:
                    continue
                for decoder_info in d['decoders']:
                    task = decoder_info['task']
                    decoder = decoder_info['path'].split("_")[2]
                    for pid, pipeline in pipelines.items():
                        if (pipeline['task'] == task and 
                            pipeline['backbone'] == backbone and 
                            pipeline['decoder'] == decoder):
                            if decoder_info['type'] == 'classification':
                                acc += metric[pid]
                                acc_count += 1
                            elif decoder_info['type'] == 'regression':
                                mae += metric[pid]
                                mae_count += 1
                            break
            
            sched_data[sched]['acc'].append(acc / acc_count if acc_count > 0 else 0)
            sched_data[sched]['mae'].append(mae / mae_count if mae_count > 0 else 0)
    
    x = np.arange(len(req_rates))
    width = 0.35 / len(schedulers)
    
    # Plot 1: Accuracy
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, sched in enumerate(schedulers):
        offset = (idx - len(schedulers) / 2 + 0.5) * width
        ax.bar(x + offset, sched_data[sched]['acc'], width,
               label=SCHEDULER_LABELS.get(sched, sched),
               color=SCHEDULER_COLORS.get(sched, "gray"), alpha=0.8, edgecolor="black", linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(rr) for rr in req_rates])
    ax.set_xlabel('Request Rate (req/s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy across Schedulers and Request Rates', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"accuracy_comparison.{fmt}"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: accuracy_comparison.{fmt}")
    
    # Plot 2: MAE
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, sched in enumerate(schedulers):
        offset = (idx - len(schedulers) / 2 + 0.5) * width
        ax.bar(x + offset, sched_data[sched]['mae'], width,
               label=SCHEDULER_LABELS.get(sched, sched),
               color=SCHEDULER_COLORS.get(sched, "gray"), alpha=0.8, edgecolor="black", linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(rr) for rr in req_rates])
    ax.set_xlabel('Request Rate (req/s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAE (Mean Absolute Error)', fontsize=12, fontweight='bold')
    ax.set_title('MAE across Schedulers and Request Rates', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"mae_comparison.{fmt}"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: mae_comparison.{fmt}")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare schedulers for FMaaS evaluation")
    parser.add_argument("--exp-dir", type=str, default="experiments/baselines/results",
                        help="Base experiment directory")
    parser.add_argument("--output-dir", type=str, default="experiments/baselines/plots",
                        help="Directory to save plots (defaults to <exp-dir>/plots)")
    parser.add_argument("--schedulers", type=str, nargs="+", default=None,
                        help="Schedulers to compare (auto-discovered if not specified)")
    parser.add_argument("--req-rates", type=int, nargs="+", default=None,
                        help="Request rates to compare (auto-discovered if not specified)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to user_config.py for SLO definitions")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf"],
                        help="Output image format")
    args = parser.parse_args()

    exp_dir = args.exp_dir
    output_dir = args.output_dir
    if not os.path.isdir(exp_dir):
        print(f"Error: experiment directory '{exp_dir}' not found.")
        return

    # Discover or use provided schedulers/rates
    schedulers = args.schedulers or discover_schedulers(exp_dir)
    req_rates = args.req_rates or discover_req_rates(exp_dir, schedulers)

    if not schedulers:
        print("No scheduler results found. Run experiments first.")
        return
    if not req_rates:
        print("No request rate directories found. Run experiments first.")
        return

    # Load SLOs
    task_slos = load_task_slos(args.config)

    print(f"Comparing schedulers: {schedulers}")
    print(f"Request rates: {req_rates}")
    print(f"Task SLOs: {task_slos}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate all plots
    print(f"\nGenerating plots in {output_dir}/ ...")

    plot_slo_violation_rate(exp_dir, schedulers, req_rates, task_slos, output_dir, args.format)
    plot_tail_latency(exp_dir, schedulers, req_rates, output_dir, args.format)
    plot_backbone_memory(exp_dir, schedulers, req_rates, output_dir, args.format)
    plot_num_deployments(exp_dir, schedulers, req_rates, output_dir, args.format)
    plot_throughput(exp_dir, schedulers, req_rates, output_dir, args.format)
    plot_response_time_breakdown(exp_dir, schedulers, req_rates, output_dir, args.format)
    plot_latency_boxplot(exp_dir, schedulers, req_rates, output_dir, args.format)
    plot_per_task_slo(exp_dir, schedulers, req_rates, task_slos, output_dir, args.format)
    plot_metric_comparison(exp_dir, schedulers, req_rates, output_dir, args.format)
    # Print summary table
    print_summary_table(exp_dir, schedulers, req_rates, task_slos)

    print(f"\nDone! All plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
