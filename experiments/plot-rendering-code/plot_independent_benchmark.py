#!/usr/bin/env python3
"""
Plot script for random benchmark results.
Creates visualizations of similarity measures for two experiments:
1. Varying N (number of points) with fixed D (dimensionality)
2. Varying D (dimensionality) with fixed N (number of points)

Shows similarity scores only (no timing analysis).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from pathlib import Path
import argparse

# Set style for scientific publication (larger fonts for paper-ready figures)
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 28
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28
plt.rcParams['legend.fontsize'] = 32
plt.rcParams['figure.titlesize'] = 30

def find_csv_files(results_dir: Path) -> list[Path]:
    """Find all CSV files in the benchmark_random results directory."""
    csv_files = list(results_dir.glob("*.csv"))
    return sorted(csv_files)  # Sort alphabetically

def load_and_process_data(csv_path: str) -> pd.DataFrame:
    """Load and process the random benchmark data."""
    df = pd.read_csv(csv_path)
    
    # Extract measure names from column headers
    score_columns = [col for col in df.columns if col.endswith('_score')]
    
    # The measure names are already clean from baselines.py
    # Just extract the base name and use as display name
    measure_names = {}
    for col in score_columns:
        measure_name = col.replace('_score', '')
        # Use the measure name directly since they come from BASELINE_MEASURES keys
        measure_names[col] = measure_name
    
    return df, score_columns, measure_names

def create_varying_n_plot(df: pd.DataFrame, score_columns: list, measure_names: dict, output_dir: Path):
    return  # Deprecated: Only combined plot is produced

def create_varying_d_plot(df: pd.DataFrame, score_columns: list, measure_names: dict, output_dir: Path):
    return  # Deprecated: Only combined plot is produced

def create_combined_plot(df: pd.DataFrame, score_columns: list, measure_names: dict, output_dir: Path):
    """Create a combined plot with both varying N and varying D experiments."""
    n_data = df[df['experiment'] == 'varying_n']
    d_data = df[df['experiment'] == 'varying_d']
    
    if n_data.empty or d_data.empty:
        print("Cannot create combined plot: missing data for one or both experiments")
        return
    
    # Create figure with two subplots sharing y-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(23, 7), sharey=True)
    
    # Use logarithmic x-axis for both subplots (base 2 aligns with multiplicative factor 2)
    ax1.set_xscale('log', base=2)
    ax2.set_xscale('log', base=2)
    
    # Use matplotlib's tab10 colormap (standard for scientific publications)
    colors = plt.cm.tab10.colors
    
    # Distinct markers for better visibility
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Different line styles for additional distinguishability
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    
    # Collect legend handles (only once since both plots use same measures)
    from matplotlib.lines import Line2D
    legend_handles = []

    fixed_d = n_data['dim'].unique()[0]
    
    # Compact tick formatter for x-axes
    def compact_tick_formatter(value, _pos):
        try:
            v = float(value)
        except Exception:
            return str(value)
        if v >= 1000:
            k = v / 1000.0
            # Use no trailing .0 (e.g., 1k) and 1 decimal when needed (e.g., 1.6k)
            return f"{k:.1f}k".replace(".0k", "k")
        return f"{int(v)}" if v.is_integer() else f"{v:g}"
    formatter = FuncFormatter(compact_tick_formatter)

    # ===== LEFT PLOT: Varying N =====
    avg_df_n = n_data.groupby('n_points')[score_columns].mean().reset_index()
    std_df_n = n_data.groupby('n_points')[score_columns].std().reset_index()
    # Set xticks to exactly the considered N values and format as integers
    n_ticks = sorted(n_data['n_points'].unique())
    if len(n_ticks) > 0:
        ax1.set_xticks(n_ticks)
        ax1.xaxis.set_major_formatter(formatter)
    
    # First pass: plot lines and error bars (no markers)
    for i, col in enumerate(score_columns):
        if col in avg_df_n.columns:
            valid_mask = ~avg_df_n[col].isna()
            if valid_mask.any():
                x_vals = avg_df_n.loc[valid_mask, 'n_points']
                y_vals = avg_df_n.loc[valid_mask, col]
                y_err = std_df_n.loc[valid_mask, col] if col in std_df_n.columns else None
                
                ax1.plot(x_vals, y_vals, 
                           linestyle=linestyles[i % len(linestyles)],
                           linewidth=5.6,
                           color=colors[i % len(colors)], 
                           alpha=0.9, 
                           zorder=2)
                
                # Create custom legend handle (only once)
                if not legend_handles or i >= len(legend_handles):
                    legend_handles.append(Line2D([0], [0], 
                                                color=colors[i % len(colors)],
                                                linestyle=linestyles[i % len(linestyles)],
                                                linewidth=2.5,
                                                marker=markers[i % len(markers)],
                                                markersize=7,
                                                markeredgewidth=1.5,
                                                markeredgecolor='white',
                                                label=measure_names[col]))
    
    # Second pass: plot markers on top
    for i, col in enumerate(score_columns):
        if col in avg_df_n.columns:
            valid_mask = ~avg_df_n[col].isna()
            if valid_mask.any():
                x_vals = avg_df_n.loc[valid_mask, 'n_points']
                y_vals = avg_df_n.loc[valid_mask, col]
                marker_offset = i % 3
                
                ax1.plot(x_vals, y_vals,
                       marker=markers[i % len(markers)], 
                       linestyle='',
                       markersize=28,
                       color=colors[i % len(colors)],
                       markeredgewidth=2.8, markeredgecolor='white',
                       markevery=(marker_offset, 3), zorder=10)
    
    ax1.set_xlabel('Number of Data Points (N)', fontsize=30, fontweight='bold')
    ax1.set_ylabel('Similarity Score', fontsize=30, fontweight='bold')
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=4)
    ax1.text(0.02, 0.98, f'D={fixed_d}', transform=ax1.transAxes, 
             fontsize=30, fontweight='bold', va='top', ha='left')
    
    fixed_n = d_data['n_points'].unique()[0]

    # ===== RIGHT PLOT: Varying D =====
    avg_df_d = d_data.groupby('dim')[score_columns].mean().reset_index()
    std_df_d = d_data.groupby('dim')[score_columns].std().reset_index()
    # Set xticks to exactly the considered D values and format as integers
    d_ticks = sorted(d_data['dim'].unique())
    if len(d_ticks) > 0:
        ax2.set_xticks(d_ticks)
        ax2.xaxis.set_major_formatter(formatter)
    
    # First pass: plot lines and error bars (no markers)
    for i, col in enumerate(score_columns):
        if col in avg_df_d.columns:
            valid_mask = ~avg_df_d[col].isna()
            if valid_mask.any():
                x_vals = avg_df_d.loc[valid_mask, 'dim']
                y_vals = avg_df_d.loc[valid_mask, col]
                y_err = std_df_d.loc[valid_mask, col] if col in std_df_d.columns else None
                
                ax2.plot(x_vals, y_vals,
                           linestyle=linestyles[i % len(linestyles)],
                           linewidth=5.6,
                           color=colors[i % len(colors)], 
                           alpha=0.9, 
                           zorder=2)
    
    # Second pass: plot markers on top
    for i, col in enumerate(score_columns):
        if col in avg_df_d.columns:
            valid_mask = ~avg_df_d[col].isna()
            if valid_mask.any():
                x_vals = avg_df_d.loc[valid_mask, 'dim']
                y_vals = avg_df_d.loc[valid_mask, col]
                marker_offset = i % 3
                
                ax2.plot(x_vals, y_vals,
                       marker=markers[i % len(markers)], 
                       linestyle='',
                       markersize=28,
                       color=colors[i % len(colors)],
                       markeredgewidth=2.8, markeredgecolor='white',
                       markevery=(marker_offset, 3), zorder=10)
    
    ax2.set_xlabel('Dimensionality (D)', fontsize=30, fontweight='bold')
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=4)
    ax2.text(0.02, 0.98, f'N={fixed_n}', transform=ax2.transAxes, 
             fontsize=30, fontweight='bold', va='top', ha='left')
    
    # Set y-limits
    ax1.set_ylim(-0.05, 1.05)
    
    # Create a single combined legend anchored to the right of the second subplot
    legend = ax2.legend(handles=legend_handles,
                        loc='center left', bbox_to_anchor=(1.01, 0.5),
                        borderaxespad=0.0,
                        fontsize=28, framealpha=0.95, edgecolor='black',
                        fancybox=False, shadow=False, ncol=1)
    # Increase legend handle sizes
    for lh in legend.legend_handles:
        lh.set_linewidth(6.4)
        lh.set_markersize(28)
    
    # Keep a small right margin so the legend hugs the plots without wasting space
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(output_dir / 'independent_benchmark.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot independent benchmark results')
    parser.add_argument('--output-dir', default='../../plots', 
                       help='Output directory for plots (default: ../../plots)')
    
    args = parser.parse_args()
    
    # Find results directory
    results_dir = Path(__file__).parent.parent / 'results' / 'benchmark_independent_representations'
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run the benchmark_independent_representations.py script first to generate results.")
        return
    
    # Find CSV files
    csv_files = find_csv_files(results_dir)
    
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        print("Run the benchmark_independent_representations.py script first to generate results.")
        return
    
    # Use first CSV file (alphabetically)
    csv_path = csv_files[0]
    print(f"Using CSV file: {csv_path.name}")
    
    # Load and process data
    df, score_columns, measure_names = load_and_process_data(csv_path)
    
    # Create output directory
    if args.output_dir.startswith('../'):
        output_dir = Path(__file__).parent / args.output_dir
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Creating plots for independent benchmark results...")
    print(f"Data shape: {df.shape}")
    print(f"Experiments found: {df['experiment'].unique().tolist()}")
    print(f"Measures found: {list(measure_names.values())}")
    
    # Only create combined plot
    if 'varying_n' in df['experiment'].values and 'varying_d' in df['experiment'].values:
        print("Creating combined plot...")
        create_combined_plot(df, score_columns, measure_names, output_dir)
    else:
        print("Cannot create combined plot: both experiments required (varying_n and varying_d).")
    
    print(f"\nPlots saved to: {output_dir}")
    print("Generated files:")
    if 'varying_n' in df['experiment'].values and 'varying_d' in df['experiment'].values:
        print("  - independent_benchmark.png: Combined plot with both experiments")

if __name__ == "__main__":
    main() 