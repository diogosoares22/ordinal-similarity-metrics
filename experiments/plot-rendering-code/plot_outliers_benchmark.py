#!/usr/bin/env python3
"""
Plot script for outliers benchmark results (random direction only).
Creates visualizations of similarity measures as a function of sigma (outlier strength).
Generates separate plots for synthetic and CIFAR-10 data sources, plus a combined plot.

Style, colors, and layout are consistent with plot_independent_benchmark.py
for use in the same academic paper.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import argparse

# Set style for scientific publication (consistent with independent benchmark plot)
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
    """Find all CSV files in the benchmark_outliers results directory."""
    csv_files = list(results_dir.glob("*.csv"))
    return sorted(csv_files)


def load_and_process_data(csv_path: str) -> pd.DataFrame:
    """Load and process the outliers benchmark data."""
    df = pd.read_csv(csv_path)

    # Extract measure names from column headers
    score_columns = [col for col in df.columns if col.endswith('_score')]

    # Use measure names directly from columns (already cleaned in benchmark script)
    measure_names = {}
    for col in score_columns:
        measure_name = col.replace('_score', '')
        measure_names[col] = measure_name

    return df, score_columns, measure_names


def compact_tick_formatter(value, _pos):
    """Compact tick formatter (same style as independent benchmark)."""
    try:
        v = float(value)
    except Exception:
        return str(value)
    if v >= 1000:
        k = v / 1000.0
        return f"{k:.1f}k".replace(".0k", "k")
    return f"{int(v)}" if v.is_integer() else f"{v:g}"


def create_sigma_plot(df: pd.DataFrame, score_columns: list, measure_names: dict, 
                      output_dir: Path, data_source: str):
    """Create a plot with sigma on the x-axis and similarity scores on the y-axis.
    
    Args:
        df: DataFrame with benchmark results
        score_columns: List of score column names
        measure_names: Dict mapping column names to display names
        output_dir: Directory to save the plot
        data_source: Either 'synthetic' or 'cifar10'
    """
    # Filter data to the specific data source
    if 'data_source' in df.columns:
        data = df[df['data_source'] == data_source]
    else:
        # Fallback for old format CSV files
        experiment_name = f'varying_sigma_random_{data_source}' if data_source != 'synthetic' else 'varying_sigma_random'
        data = df[df['experiment'] == experiment_name] if 'experiment' in df.columns else df
    
    if data.empty:
        print(f"No data found for {data_source} data source.")
        return False

    # Aggregate by sigma
    avg_df = data.groupby('sigma')[score_columns].mean().reset_index()
    std_df = data.groupby('sigma')[score_columns].std().reset_index()

    fixed_n = data['n_points'].unique()[0]
    fixed_d = data['dim'].unique()[0]
    fixed_k = data['k_outliers'].unique()[0] if 'k_outliers' in data.columns else None

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    # Use symmetric log so 0 is allowed; beyond |1| it's log base 2.
    # This makes distances 1→2 and 2→4 equal while keeping 0 on the axis.
    ax.set_xscale('symlog', base=2, linthresh=1, linscale=1)

    # Use matplotlib's tab10 colormap and consistent markers/linestyles
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']

    formatter = FuncFormatter(compact_tick_formatter)

    # Plot lines first
    from matplotlib.lines import Line2D
    legend_handles = []
    for i, col in enumerate(score_columns):
        if col in avg_df.columns:
            valid_mask = ~avg_df[col].isna()
            if valid_mask.any():
                x_vals = avg_df.loc[valid_mask, 'sigma']
                y_vals = avg_df.loc[valid_mask, col]
                y_err = None
                if col in std_df.columns:
                    y_err = std_df.loc[valid_mask, col]
                ax.plot(x_vals, y_vals,
                        linestyle=linestyles[i % len(linestyles)],
                        linewidth=5.6,
                        color=colors[i % len(colors)],
                        alpha=0.9,
                        zorder=2)

                # Standard deviation error bars
                y_err = None # dirty fix
                if y_err is not None:
                    ax.errorbar(x_vals, y_vals, yerr=y_err,
                                fmt='none', ecolor=colors[i % len(colors)],
                                elinewidth=2.2, capsize=6, alpha=0.35, zorder=3)

                # Legend handle
                legend_handles.append(Line2D([0], [0],
                                             color=colors[i % len(colors)],
                                             linestyle=linestyles[i % len(linestyles)],
                                             linewidth=2.5,
                                             marker=markers[i % len(markers)],
                                             markersize=7,
                                             markeredgewidth=1.5,
                                             markeredgecolor='white',
                                             label=measure_names[col]))

    # Overlay markers
    for i, col in enumerate(score_columns):
        if col in avg_df.columns:
            valid_mask = ~avg_df[col].isna()
            if valid_mask.any():
                x_vals = avg_df.loc[valid_mask, 'sigma']
                y_vals = avg_df.loc[valid_mask, col]
                marker_offset = i % 3
                ax.plot(x_vals, y_vals,
                        marker=markers[i % len(markers)],
                        linestyle='',
                        markersize=28,
                        color=colors[i % len(colors)],
                        markeredgewidth=2.8, markeredgecolor='white',
                        markevery=(marker_offset, 3), zorder=10)

    # Labels and grid
    ax.set_xlabel('Outlier Strength (\u03c3)', fontsize=30, fontweight='bold')
    ax.set_ylabel('Similarity Score', fontsize=30, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=4)
    ax.text(0.02, 0.36, f'N={fixed_n}', transform=ax.transAxes, 
             fontsize=30, fontweight='bold', va='top', ha='left')
    ax.text(0.02, 0.24, f'D={fixed_d}', transform=ax.transAxes, 
             fontsize=30, fontweight='bold', va='top', ha='left')
    if fixed_k is not None:
        ax.text(0.02, 0.12, f'K={fixed_k}', transform=ax.transAxes, 
                 fontsize=30, fontweight='bold', va='top', ha='left')
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    sigma_ticks = sorted([int(s) for s in data['sigma'].unique()])
    ax.set_xticks(sigma_ticks)
    ax.xaxis.set_major_formatter(formatter)

    # Legend formatting (to the right)
    legend = ax.legend(handles=legend_handles,
                       loc='center left', bbox_to_anchor=(1.01, 0.5),
                       borderaxespad=0.0,
                       fontsize=28, framealpha=0.95, edgecolor='black',
                       fancybox=False, shadow=False, ncol=1)
    for lh in legend.legend_handles:
        lh.set_linewidth(6.4)
        lh.set_markersize(28)

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    
    # Save with data source in filename
    output_filename = f'outliers_benchmark_{data_source}.png'
    plt.savefig(output_dir / output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True


def create_combined_plot(df: pd.DataFrame, score_columns: list, measure_names: dict, 
                         output_dir: Path):
    """Create a combined plot with synthetic and CIFAR-10 side by side.
    
    Args:
        df: DataFrame with benchmark results
        score_columns: List of score column names
        measure_names: Dict mapping column names to display names
        output_dir: Directory to save the plot
    """
    from matplotlib.lines import Line2D
    
    # Check if we have both data sources
    if 'data_source' not in df.columns:
        print("Combined plot requires data_source column in CSV.")
        return False
    
    data_sources = df['data_source'].unique().tolist()
    
    # Ensure we have both synthetic and cifar10
    if 'synthetic' not in data_sources or 'cifar10' not in data_sources:
        print("Combined plot requires both synthetic and cifar10 data sources.")
        return False
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(24, 6), sharey=True)
    
    # Use matplotlib's tab10 colormap and consistent markers/linestyles
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    
    formatter = FuncFormatter(compact_tick_formatter)
    
    legend_handles = []
    
    # Plot order: synthetic first, then cifar10
    plot_order = ['synthetic', 'cifar10']
    
    for ax_idx, data_source in enumerate(plot_order):
        ax = axes[ax_idx]
        data = df[df['data_source'] == data_source]
        
        if data.empty:
            continue
        
        # Aggregate by sigma
        avg_df = data.groupby('sigma')[score_columns].mean().reset_index()
        std_df = data.groupby('sigma')[score_columns].std().reset_index()
        
        fixed_n = data['n_points'].unique()[0]
        fixed_d = data['dim'].unique()[0]
        fixed_k = data['k_outliers'].unique()[0] if 'k_outliers' in data.columns else None
        
        # Use symmetric log scale
        ax.set_xscale('symlog', base=2, linthresh=1, linscale=1)
        
        # Plot lines first
        for i, col in enumerate(score_columns):
            if col in avg_df.columns:
                valid_mask = ~avg_df[col].isna()
                if valid_mask.any():
                    x_vals = avg_df.loc[valid_mask, 'sigma']
                    y_vals = avg_df.loc[valid_mask, col]
                    ax.plot(x_vals, y_vals,
                            linestyle=linestyles[i % len(linestyles)],
                            linewidth=5.6,
                            color=colors[i % len(colors)],
                            alpha=0.9,
                            zorder=2)
                    
                    # Only create legend handles from first subplot
                    if ax_idx == 0:
                        legend_handles.append(Line2D([0], [0],
                                                     color=colors[i % len(colors)],
                                                     linestyle=linestyles[i % len(linestyles)],
                                                     linewidth=2.5,
                                                     marker=markers[i % len(markers)],
                                                     markersize=7,
                                                     markeredgewidth=1.5,
                                                     markeredgecolor='white',
                                                     label=measure_names[col]))
        
        # Overlay markers
        for i, col in enumerate(score_columns):
            if col in avg_df.columns:
                valid_mask = ~avg_df[col].isna()
                if valid_mask.any():
                    x_vals = avg_df.loc[valid_mask, 'sigma']
                    y_vals = avg_df.loc[valid_mask, col]
                    marker_offset = i % 3
                    ax.plot(x_vals, y_vals,
                            marker=markers[i % len(markers)],
                            linestyle='',
                            markersize=28,
                            color=colors[i % len(colors)],
                            markeredgewidth=2.8, markeredgecolor='white',
                            markevery=(marker_offset, 3), zorder=10)
        
        # Labels and grid
        ax.set_xlabel('Outlier Strength (\u03c3)', fontsize=30, fontweight='bold')
        if ax_idx == 0:
            ax.set_ylabel('Similarity Score', fontsize=30, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=4)
        
        # Add N, D, K annotations
        ax.text(0.02, 0.36, f'N={fixed_n}', transform=ax.transAxes, 
                 fontsize=30, fontweight='bold', va='top', ha='left')
        ax.text(0.02, 0.24, f'D={fixed_d}', transform=ax.transAxes, 
                 fontsize=30, fontweight='bold', va='top', ha='left')
        if fixed_k is not None:
            ax.text(0.02, 0.12, f'K={fixed_k}', transform=ax.transAxes, 
                     fontsize=30, fontweight='bold', va='top', ha='left')
        
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        sigma_ticks = sorted([int(s) for s in data['sigma'].unique()])
        ax.set_xticks(sigma_ticks)
        ax.xaxis.set_major_formatter(formatter)
    
    # Add legend only to the second subplot (CIFAR-10)
    legend = axes[1].legend(handles=legend_handles,
                            loc='center left', bbox_to_anchor=(1.01, 0.5),
                            borderaxespad=0.0,
                            fontsize=28, framealpha=0.95, edgecolor='black',
                            fancybox=False, shadow=False, ncol=1)
    for lh in legend.legend_handles:
        lh.set_linewidth(6.4)
        lh.set_markersize(28)
    
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    
    # Save combined plot
    output_filename = 'outliers_benchmark_combined.png'
    plt.savefig(output_dir / output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Plot outliers benchmark results (random direction)')
    parser.add_argument('--output-dir', default='../../plots',
                       help='Output directory for plots (default: ../../plots)')

    args = parser.parse_args()

    # Find results directory
    results_dir = Path(__file__).parent.parent / 'results' / 'benchmark_outliers'

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run the benchmark_outliers.py script first to generate results.")
        return

    # Find CSV files
    csv_files = find_csv_files(results_dir)

    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        print("Run the benchmark_outliers.py script first to generate results.")
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

    print(f"Creating plots for outliers benchmark results...")
    print(f"Data shape: {df.shape}")
    print(f"Measures found: {list(measure_names.values())}")
    
    # Get available data sources from the CSV
    if 'data_source' in df.columns:
        data_sources = df['data_source'].unique().tolist()
    else:
        # Fallback for old format - assume synthetic only
        data_sources = ['synthetic']
    
    print(f"Data sources found: {data_sources}")
    
    generated_files = []
    
    # Generate a plot for each data source
    for data_source in data_sources:
        print(f"\nGenerating plot for {data_source} data...")
        success = create_sigma_plot(df, score_columns, measure_names, output_dir, data_source)
        if success:
            generated_files.append(f'outliers_benchmark_{data_source}.png')
    
    # Generate combined plot if we have both data sources
    if 'synthetic' in data_sources and 'cifar10' in data_sources:
        print(f"\nGenerating combined plot...")
        success = create_combined_plot(df, score_columns, measure_names, output_dir)
        if success:
            generated_files.append('outliers_benchmark_combined.png')

    print(f"\nPlots saved to: {output_dir}")
    print("Generated files:")
    for f in generated_files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
