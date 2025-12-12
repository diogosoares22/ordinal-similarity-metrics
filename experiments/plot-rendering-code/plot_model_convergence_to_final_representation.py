#!/usr/bin/env python3
"""
Plot script for model convergence to final representation.
Creates visualizations of similarity measures as a function of training epoch,
showing how representations converge to the final trained representation.
Generates plots for both CIFAR-10 and CIFAR-100 datasets, averaging across seeds.

Style, colors, and layout are consistent with other benchmark plots
for use in the same academic paper.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import argparse
import re

# Set style for scientific publication (consistent with other plots)
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 28
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28
plt.rcParams['legend.fontsize'] = 32
plt.rcParams['figure.titlesize'] = 30


def find_experiment_folders(results_dir: Path, dataset: str) -> list[Path]:
    """Find all experiment folders for a given dataset."""
    pattern = f"vit_{dataset}_*"
    folders = sorted(results_dir.glob(pattern))
    return [f for f in folders if f.is_dir()]


def load_similarity_data(folder: Path) -> pd.DataFrame | None:
    """Load similarity_to_final_epoch.csv from an experiment folder."""
    csv_path = folder / "similarity_to_final_epoch.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def extract_seed_from_folder(folder: Path) -> int | None:
    """Extract seed number from folder name."""
    match = re.search(r'seed(\d+)', folder.name)
    if match:
        return int(match.group(1))
    return None


def load_all_seeds(results_dir: Path, dataset: str) -> tuple[pd.DataFrame, int]:
    """
    Load and combine similarity data from all seeds for a dataset.
    
    Returns:
        Combined DataFrame with all seeds, and number of seeds found
    """
    folders = find_experiment_folders(results_dir, dataset)
    
    all_dfs = []
    for folder in folders:
        seed = extract_seed_from_folder(folder)
        df = load_similarity_data(folder)
        if df is not None and seed is not None:
            df['seed'] = seed
            all_dfs.append(df)
    
    if not all_dfs:
        return pd.DataFrame(), 0
    
    combined = pd.concat(all_dfs, ignore_index=True)
    n_seeds = len(all_dfs)
    return combined, n_seeds


def get_score_columns(df: pd.DataFrame) -> list[str]:
    """Get selected score columns from the DataFrame (C-CKA, C-TSI, C-QSI only)."""
    # Only include these specific columns
    target_cols = ['C-TSI', 'C-QSI', 'C-CKA', 'B-CKNNA', 'B-MutualNN']
    return target_cols


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


def create_convergence_plot(df: pd.DataFrame, score_columns: list, output_dir: Path, 
                            dataset: str, n_seeds: int):
    """Create a convergence plot for a single dataset.
    
    Args:
        df: DataFrame with similarity data (combined across seeds)
        score_columns: List of score column names to plot
        output_dir: Directory to save the plot
        dataset: Dataset name (e.g., 'cifar10', 'cifar100')
        n_seeds: Number of seeds used for averaging
    """
    from matplotlib.lines import Line2D
    
    if df.empty:
        print(f"No data found for {dataset}.")
        return False
    
    # Aggregate by epoch: mean and std across seeds
    avg_df = df.groupby('epoch')[score_columns].mean().reset_index()
    std_df = df.groupby('epoch')[score_columns].std().reset_index()
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    
    # Use matplotlib's tab10 colormap and consistent markers/linestyles
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    
    formatter = FuncFormatter(compact_tick_formatter)
    
    legend_handles = []
    
    # Plot lines first
    for i, col in enumerate(score_columns):
        if col in avg_df.columns:
            valid_mask = ~avg_df[col].isna()
            if valid_mask.any():
                x_vals = avg_df.loc[valid_mask, 'epoch']
                y_vals = avg_df.loc[valid_mask, col]
                y_err = std_df.loc[valid_mask, col] if col in std_df.columns else None
                
                ax.plot(x_vals, y_vals,
                        linestyle=linestyles[i % len(linestyles)],
                        linewidth=5.6,
                        color=colors[i % len(colors)],
                        alpha=0.9,
                        zorder=2)
                
                # Standard deviation shading
                if y_err is not None and not y_err.isna().all():
                    ax.fill_between(x_vals, y_vals - y_err, y_vals + y_err,
                                    color=colors[i % len(colors)], alpha=0.15, zorder=1)
                
                # Legend handle
                legend_handles.append(Line2D([0], [0],
                                             color=colors[i % len(colors)],
                                             linestyle=linestyles[i % len(linestyles)],
                                             linewidth=2.5,
                                             marker=markers[i % len(markers)],
                                             markersize=7,
                                             markeredgewidth=1.5,
                                             markeredgecolor='white',
                                             label=col))
    
    # Overlay markers
    for i, col in enumerate(score_columns):
        if col in avg_df.columns:
            valid_mask = ~avg_df[col].isna()
            if valid_mask.any():
                x_vals = avg_df.loc[valid_mask, 'epoch']
                y_vals = avg_df.loc[valid_mask, col]
                marker_offset = i % 3
                # Use markevery to not overcrowd with markers (show ~5 markers)
                n_points = len(x_vals)
                markevery = max(1, n_points // 5)
                ax.plot(x_vals, y_vals,
                        marker=markers[i % len(markers)],
                        linestyle='',
                        markersize=20,
                        color=colors[i % len(colors)],
                        markeredgewidth=2.0, markeredgecolor='white',
                        markevery=(marker_offset * (markevery // 3), markevery), zorder=10)
    
    # Labels and grid
    ax.set_xlabel('Epoch', fontsize=30, fontweight='bold')
    ax.set_ylabel('Similarity Score', fontsize=30, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=4)
    
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(-5, avg_df['epoch'].max() + 5)
    
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
    
    # Save with dataset in filename
    output_filename = f'model_convergence_{dataset}.png'
    plt.savefig(output_dir / output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True


def create_combined_plot(data_by_dataset: dict, score_columns: list, output_dir: Path):
    """Create a combined plot with CIFAR-10 and CIFAR-100 side by side.
    
    Args:
        data_by_dataset: Dict mapping dataset name to (df, n_seeds) tuple
        score_columns: List of score column names to plot
        output_dir: Directory to save the plot
    """
    from matplotlib.lines import Line2D
    
    # Check if we have both datasets
    if 'cifar10' not in data_by_dataset or 'cifar100' not in data_by_dataset:
        print("Combined plot requires both CIFAR-10 and CIFAR-100 data.")
        return False
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(24, 6), sharey=True)
    
    # Use matplotlib's tab10 colormap and consistent markers/linestyles
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    
    formatter = FuncFormatter(compact_tick_formatter)
    
    legend_handles = []
    
    # Plot order
    plot_order = ['cifar10', 'cifar100']
    
    for ax_idx, dataset in enumerate(plot_order):
        ax = axes[ax_idx]
        df, n_seeds = data_by_dataset[dataset]
        
        if df.empty:
            continue
        
        # Aggregate by epoch
        avg_df = df.groupby('epoch')[score_columns].mean().reset_index()
        std_df = df.groupby('epoch')[score_columns].std().reset_index()
        
        # Plot lines first
        for i, col in enumerate(score_columns):
            if col in avg_df.columns:
                valid_mask = ~avg_df[col].isna()
                if valid_mask.any():
                    x_vals = avg_df.loc[valid_mask, 'epoch']
                    y_vals = avg_df.loc[valid_mask, col]
                    y_err = std_df.loc[valid_mask, col] if col in std_df.columns else None
                    
                    ax.plot(x_vals, y_vals,
                            linestyle=linestyles[i % len(linestyles)],
                            linewidth=5.6,
                            color=colors[i % len(colors)],
                            alpha=0.9,
                            zorder=2)
                    
                    # Standard deviation shading
                    if y_err is not None and not y_err.isna().all():
                        ax.fill_between(x_vals, y_vals - y_err, y_vals + y_err,
                                        color=colors[i % len(colors)], alpha=0.15, zorder=1)
                    
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
                                                     label=col))
        
        # Overlay markers
        for i, col in enumerate(score_columns):
            if col in avg_df.columns:
                valid_mask = ~avg_df[col].isna()
                if valid_mask.any():
                    x_vals = avg_df.loc[valid_mask, 'epoch']
                    y_vals = avg_df.loc[valid_mask, col]
                    marker_offset = i % 3
                    n_points = len(x_vals)
                    markevery = max(1, n_points // 5)  # Show ~5 markers
                    ax.plot(x_vals, y_vals,
                            marker=markers[i % len(markers)],
                            linestyle='',
                            markersize=20,
                            color=colors[i % len(colors)],
                            markeredgewidth=2.0, markeredgecolor='white',
                            markevery=(marker_offset * (markevery // 3), markevery), zorder=10)
        
        # Labels and grid
        ax.set_xlabel('Epoch', fontsize=30, fontweight='bold')
        if ax_idx == 0:
            ax.set_ylabel('Similarity Score', fontsize=30, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=4)
        
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlim(-5, avg_df['epoch'].max() + 5)
    
    # Add legend only to the second subplot
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
    output_filename = 'model_convergence_combined.png'
    plt.savefig(output_dir / output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True


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
    
    # Load data for both datasets
    datasets = ['cifar10', 'cifar100']
    data_by_dataset = {}
    score_columns = []
    
    for dataset in datasets:
        df, n_seeds = load_all_seeds(results_dir, dataset)
        if not df.empty:
            data_by_dataset[dataset] = (df, n_seeds)
            score_cols = get_score_columns(df)
            score_columns = score_cols
            print(f"  {dataset.upper()}: {n_seeds} seeds, {len(df)} rows")
        else:
            print(f"  {dataset.upper()}: No data found")
    
    if not data_by_dataset:
        print("No data found for any dataset.")
        return
    
    # Use common score columns across datasets (ordered)
    print(f"Measures found: {score_columns}")
    
    generated_files = []
    
    # Generate individual plots for each dataset
    for dataset, (df, n_seeds) in data_by_dataset.items():
        print(f"\nGenerating plot for {dataset.upper()}...")
        # Get score columns for this specific dataset
        dataset_score_cols = score_columns
        success = create_convergence_plot(df, dataset_score_cols, output_dir, dataset, n_seeds)
        if success:
            generated_files.append(f'model_convergence_{dataset}.png')
    
    # Generate combined plot if we have both datasets
    if 'cifar10' in data_by_dataset and 'cifar100' in data_by_dataset:
        print(f"\nGenerating combined plot...")
        common_cols = score_columns
        success = create_combined_plot(data_by_dataset, common_cols, output_dir)
        if success:
            generated_files.append('model_convergence_combined.png')

    print(f"\nPlots saved to: {output_dir}")
    print("Generated files:")
    for f in generated_files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()

