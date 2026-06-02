#!/usr/bin/env python3
"""
Plot script for model convergence to final representation.
Creates a combined visualization with TSI, QSI, and CKA showing how representations 
converge to final representations for both same-seed and distinct-seed comparisons.

Only plots CIFAR-10 results, averaging across seeds.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from pathlib import Path
import argparse
import re

# Set style for scientific publication
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 28
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28
plt.rcParams['legend.fontsize'] = 32
plt.rcParams['figure.titlesize'] = 30

# Color scheme for same-seed vs distinct-seed
COLORS = {
    'TSI': {'same': '#1b7837', 'distinct': '#762a83'},  # Green vs Purple
    'QSI': {'same': '#2166ac', 'distinct': '#b2182b'},  # Blue vs Red
    'CKA': {'same': '#d95f02', 'distinct': '#7570b3'},  # Orange vs Violet
}

# Display names for metrics
METRIC_DISPLAY = {
    'C-TSI': 'TSI',
    'C-QSI': 'QSI', 
    'C-CKA': 'CKA',
}


def find_experiment_folders(results_dir: Path, dataset: str) -> list[Path]:
    """Find all experiment folders for a given dataset."""
    pattern = f"vit_{dataset}_*"
    folders = sorted(results_dir.glob(pattern))
    return [f for f in folders if f.is_dir()]


def extract_seed_from_folder(folder: Path) -> int | None:
    """Extract seed number from folder name."""
    match = re.search(r'seed(\d+)', folder.name)
    if match:
        return int(match.group(1))
    return None


def load_same_seed_data(results_dir: Path) -> tuple[pd.DataFrame, int]:
    """
    Load same-seed convergence data from all seeds.
    Uses similarity_to_final_epoch.csv from each seed folder.
    
    Returns:
        Combined DataFrame with all seeds, and number of seeds found
    """
    folders = find_experiment_folders(results_dir, 'cifar10')
    
    all_dfs = []
    for folder in folders:
        seed = extract_seed_from_folder(folder)
        csv_path = folder / "similarity_to_final_epoch.csv"
        if csv_path.exists() and seed is not None:
            df = pd.read_csv(csv_path)
            df['seed'] = seed
            all_dfs.append(df)
    
    if not all_dfs:
        return pd.DataFrame(), 0
    
    combined = pd.concat(all_dfs, ignore_index=True)
    n_seeds = len(all_dfs)
    return combined, n_seeds


def load_cross_seed_data(results_dir: Path) -> tuple[pd.DataFrame, int]:
    """
    Load distinct-seed convergence data from cross_seed_similarity CSV.
    Only keeps rows where seed_final != seed_trajectory.
    
    Returns:
        DataFrame with cross-seed comparisons and number of unique seed pairs
    """
    # Find the cross_seed_similarity file (should be in seed0 folder)
    seed0_folder = results_dir / "vit_cifar10_lr0.0001_epochs200_seed0"
    
    # Try to find the cross_seed file
    cross_seed_files = list(seed0_folder.glob("cross_seed_similarity_*.csv"))
    
    if not cross_seed_files:
        print(f"No cross_seed_similarity file found in {seed0_folder}")
        return pd.DataFrame(), 0
    
    csv_path = cross_seed_files[0]
    df = pd.read_csv(csv_path)
    
    # Filter to only distinct seed comparisons
    df = df[df['seed_final'] != df['seed_trajectory']].copy()
    
    # Rename epoch_trajectory to epoch for consistency
    df = df.rename(columns={'epoch_trajectory': 'epoch'})
    
    # Count unique seed pairs
    n_pairs = len(df[['seed_final', 'seed_trajectory']].drop_duplicates())
    
    return df, n_pairs


def aggregate_same_seed(df: pd.DataFrame, metric: str) -> tuple[pd.Series, pd.Series]:
    """Aggregate same-seed data by epoch, returning mean and std."""
    grouped = df.groupby('epoch')[metric]
    return grouped.mean(), grouped.std()


def aggregate_cross_seed(df: pd.DataFrame, metric: str) -> tuple[pd.Series, pd.Series]:
    """Aggregate cross-seed data by epoch, returning mean and std across all seed pairs."""
    grouped = df.groupby('epoch')[metric]
    return grouped.mean(), grouped.std()


def compact_tick_formatter(value, _pos):
    """Compact tick formatter for axes."""
    try:
        v = float(value)
    except Exception:
        return str(value)
    if v >= 1000:
        k = v / 1000.0
        return f"{k:.1f}k".replace(".0k", "k")
    return f"{int(v)}" if v.is_integer() else f"{v:g}"


def create_combined_plot(same_seed_df: pd.DataFrame, cross_seed_df: pd.DataFrame,
                         metrics: list, output_dir: Path):
    """
    Create a combined convergence plot with all metrics sharing the x-axis.
    
    Args:
        same_seed_df: DataFrame with same-seed similarity data
        cross_seed_df: DataFrame with cross-seed similarity data
        metrics: List of metric column names (e.g., ['C-TSI', 'C-QSI', 'C-CKA'])
        output_dir: Directory to save the plot
    """
    n_metrics = len(metrics)
    
    # Create figure with shared y-axis (now horizontal)
    fig, axes = plt.subplots(1, n_metrics, figsize=(32, 7), sharey=True)
    
    # Ensure axes is always a list
    if n_metrics == 1:
        axes = [axes]
    
    # Get max epoch for x-axis limits
    max_epoch = 0
    if not same_seed_df.empty:
        max_epoch = max(max_epoch, same_seed_df['epoch'].max())
    if not cross_seed_df.empty:
        max_epoch = max(max_epoch, cross_seed_df['epoch'].max())
    
    for idx, (ax, metric) in enumerate(zip(axes, metrics)):
        metric_short = METRIC_DISPLAY.get(metric, metric)
        colors = COLORS.get(metric_short, {'same': '#1f77b4', 'distinct': '#ff7f0e'})
        
        # Plot same-seed convergence
        if not same_seed_df.empty and metric in same_seed_df.columns:
            same_mean, same_std = aggregate_same_seed(same_seed_df, metric)
            epochs = same_mean.index
            
            # Plot line
            ax.plot(epochs, same_mean,
                    linestyle='-',
                    linewidth=5.6,
                    color=colors['same'],
                    alpha=0.9,
                    zorder=2)
            
            # Std shading
            if same_std is not None and not same_std.isna().all():
                ax.fill_between(epochs, same_mean - same_std, same_mean + same_std,
                               color=colors['same'], alpha=0.15, zorder=1)
            
            # Add markers
            n_points = len(epochs)
            markevery = max(1, n_points // 6)
            ax.plot(epochs, same_mean,
                    marker='o',
                    linestyle='',
                    markersize=28,
                    color=colors['same'],
                    markeredgewidth=2.8, 
                    markeredgecolor='white',
                    markevery=markevery, 
                    zorder=10)
        
        # Plot distinct-seed convergence
        if not cross_seed_df.empty and metric in cross_seed_df.columns:
            cross_mean, cross_std = aggregate_cross_seed(cross_seed_df, metric)
            epochs = cross_mean.index
            
            # Plot line
            ax.plot(epochs, cross_mean,
                    linestyle='--',
                    linewidth=5.6,
                    color=colors['distinct'],
                    alpha=0.9,
                    zorder=2)
            
            # Std shading
            if cross_std is not None and not cross_std.isna().all():
                ax.fill_between(epochs, cross_mean - cross_std, cross_mean + cross_std,
                               color=colors['distinct'], alpha=0.15, zorder=1)
            
            # Add markers
            n_points = len(epochs)
            markevery = max(1, n_points // 6)
            ax.plot(epochs, cross_mean,
                    marker='s',
                    linestyle='',
                    markersize=28,
                    color=colors['distinct'],
                    markeredgewidth=2.8, 
                    markeredgecolor='white',
                    markevery=(markevery // 2, markevery),  # Offset markers
                    zorder=10)
        
        # Set title as the metric name
        ax.set_title(metric_short, fontsize=30, fontweight='bold')
        
        # Y-axis label only on the first subplot
        if idx == 0:
            ax.set_ylabel('Similarity Score', fontsize=30, fontweight='bold')
        
        # X-axis label on all subplots
        ax.set_xlabel('Epoch', fontsize=30, fontweight='bold')
        
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=4)
        
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlim(-5, max_epoch + 5)
        
        # Add metric-specific legend in each subplot showing the colors
        metric_handles = [
            Line2D([0], [0], color=colors['same'], linestyle='-', linewidth=5.6,
                   marker='o', markersize=20, markeredgewidth=2, markeredgecolor='white',
                   label='Same Seed'),
            Line2D([0], [0], color=colors['distinct'], linestyle='--', linewidth=5.6,
                   marker='s', markersize=20, markeredgewidth=2, markeredgecolor='white',
                   label='Distinct Seed'),
        ]
        ax.legend(handles=metric_handles, loc='lower right', fontsize=26,
                  framealpha=0.95, edgecolor='black', fancybox=False)
    
    plt.tight_layout()
    
    # Save
    output_filename = 'model_convergence_combined_cifar10.png'
    plt.savefig(output_dir / output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Generated: {output_filename}")
    return output_filename


def main():
    parser = argparse.ArgumentParser(description='Plot model convergence to final representation')
    parser.add_argument('--output-dir', default='../../plots',
                       help='Output directory for plots (default: ../../plots)')
    parser.add_argument('--results-dir', default=None,
                       help='Results directory (default: experiments/results/train_model_and_compare_representations)')

    args = parser.parse_args()

    # Find results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(__file__).parent.parent / 'results' / 'train_model_and_compare_representations'

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run the train_model_and_compare_representations.py script first to generate results.")
        return

    # Create output directory
    if args.output_dir.startswith('../'):
        output_dir = Path(__file__).parent / args.output_dir
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Loading data from: {results_dir}")
    
    # Load same-seed data (CIFAR-10 only)
    same_seed_df, n_same_seeds = load_same_seed_data(results_dir)
    if same_seed_df.empty:
        print("No same-seed data found for CIFAR-10.")
    else:
        print(f"  Same-seed: {n_same_seeds} seeds, {len(same_seed_df)} rows")
    
    # Load cross-seed data
    cross_seed_df, n_cross_pairs = load_cross_seed_data(results_dir)
    if cross_seed_df.empty:
        print("No cross-seed data found.")
    else:
        print(f"  Cross-seed: {n_cross_pairs} seed pairs, {len(cross_seed_df)} rows")
    
    if same_seed_df.empty and cross_seed_df.empty:
        print("No data found. Exiting.")
        return
    
    # Metrics to plot (in order: TSI, QSI, CKA)
    metrics = ['C-TSI', 'C-QSI', 'C-CKA']
    
    print(f"\nGenerating combined plot for metrics: {[METRIC_DISPLAY[m] for m in metrics]}")
    
    filename = create_combined_plot(same_seed_df, cross_seed_df, metrics, output_dir)

    print(f"\nPlot saved to: {output_dir}")
    print(f"  - {filename}")


if __name__ == "__main__":
    main()
