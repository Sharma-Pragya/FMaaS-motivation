#!/usr/bin/env python3
"""Plot placement results with confidence intervals.

Usage:
    python plot.py [output_dir] [--mode fixed-n|admission] [--output plot.pdf]

Examples:
    # Plot specific run
    python plot.py outputs/run_20260513_120000 --mode admission
    
    # Plot all runs in outputs/
    python plot.py --mode admission
    
    # Save to file
    python plot.py outputs/run_20260513_120000 --output results.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


def plot_placement_results(summary_path: Path, mode: str, output_path: Path | None = None) -> None:
    """Plot placement results with confidence intervals."""
    
    with open(summary_path, 'r') as f:
        data = json.load(f)
    
    scenarios = data['scenarios']
    
    # Extract data by regime with fixed order
    all_regimes = set(s['regime'] for s in scenarios)
    regime_order = {'low': 0, 'medium': 1, 'high': 2}
    regimes = sorted(all_regimes, key=lambda r: regime_order.get(r, 999))
    condition_names = {'fmaas': 'FMVisor', 'no_sharing': 'BE'}
    
    # Prepare data for plotting
    placed_counts = {regime: {} for regime in regimes}
    
    # Determine which metric to use based on mode
    metric_key = 'admitted_before_failure' if mode == 'admission' else 'placed_count'
    
    for scenario in scenarios:
        regime = scenario['regime']
        conditions = scenario['conditions']
        
        for condition_key, condition_data in conditions.items():
            if condition_key not in condition_names:
                continue
            
            if metric_key not in condition_data:
                print(f"Warning: {metric_key} not found in {condition_key} data")
                continue
            
            metric = condition_data[metric_key]
            placed_counts[regime][condition_key] = {
                'mean': metric['mean'],
                'ci_low': metric['ci95_low'],
                'ci_high': metric['ci95_high'],
                'std': metric['std'],
            }
    
    # Create plot
    fig, ax = plt.subplots(figsize=(4.0, 2.5), dpi=150)
    
    x_positions = np.arange(len(regimes))
    width = 0.35
    
    conditions = ['fmaas', 'no_sharing']
    colors = {'fmaas': '#E06C75', 'no_sharing': '#888888'}
    
    for i, condition in enumerate(conditions):
        means = []
        errors = []
        
        for regime in regimes:
            if condition in placed_counts[regime]:
                data_point = placed_counts[regime][condition]
                means.append(data_point['mean'])
                # Calculate error bar size (half-width of CI)
                error = data_point['mean'] - data_point['ci_low']
                errors.append(error)
            else:
                means.append(np.nan)
                errors.append(0)
        
        offset = width * (i - 0.5)
        ax.bar(x_positions + offset, means, width, 
               label=condition_names[condition],
               color=colors[condition],
               capsize=5,
               error_kw={'linewidth': 1.5},
               yerr=errors)
    
    # Formatting
    ax.set_ylabel('#Task Placed', fontsize=12, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([r.capitalize() for r in regimes], fontsize=12)
    ax.set_yscale('log')
    ax.legend(fontsize=10, frameon=True, fancybox=False, edgecolor='black')
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', nargs='?', default=None,
                       help='Path to experiment output directory. If not provided, plots all runs in outputs/')
    parser.add_argument('--mode', choices=['fixed-n', 'admission'], 
                       default='admission',
                       help='Experiment mode (default: admission)')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output plot path (e.g., plot.pdf, plot.png). Only used with specific output_dir')
    args = parser.parse_args()
    
    # Determine which directories to process
    if args.output_dir:
        output_dirs = [Path(args.output_dir)]
    else:
        # Find all run directories
        if not DEFAULT_OUTPUT_ROOT.exists():
            print(f"Error: outputs directory not found at {DEFAULT_OUTPUT_ROOT}")
            return 1
        output_dirs = sorted([d for d in DEFAULT_OUTPUT_ROOT.iterdir() if d.is_dir()])
        if not output_dirs:
            print(f"Error: No run directories found in {DEFAULT_OUTPUT_ROOT}")
            return 1
        print(f"Found {len(output_dirs)} run directories")
    
    # Process each directory
    for output_dir in output_dirs:
        if args.mode == 'admission':
            summary_file = output_dir / 'admission_aggregate_summary.json'
        else:
            summary_file = output_dir / 'aggregate_summary.json'
        
        if not summary_file.exists():
            print(f"Skipping {output_dir.name}: Summary file not found")
            continue
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            output_path = output_dir / f'placement_results_{args.mode}.pdf'
        
        print(f"Processing {output_dir.name}...")
        plot_placement_results(summary_file, args.mode, Path(output_path))
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
