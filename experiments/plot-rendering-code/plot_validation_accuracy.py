#!/usr/bin/env python3
"""
Plot script for validation accuracy over training epochs.
Creates visualizations of validation accuracy as a function of epoch number
from training logs, plotting all seeds together with a legend.

Style, colors, and layout are consistent with other benchmark plots
for use in the same academic paper.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
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
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['figure.titlesize'] = 30


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


def load_training_data(csv_path: Path) -> pd.DataFrame:
    """Load training logs from CSV file."""
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def load_all_seeds(results_dir: Path, dataset: str) -> dict[int, pd.DataFrame]:
    """Load training data from all seeds for a dataset.
    
    Returns:
        Dict mapping seed number to DataFrame
    """
    folders = find_experiment_folders(results_dir, dataset)
    
    seed_data = {}
    for folder in folders:
        seed = extract_seed_from_folder(folder)
        csv_path = folder / "training_logs.csv"
        df = load_training_data(csv_path)
        if not df.empty and seed is not None:
            seed_data[seed] = df
    
    return seed_data


def create_validation_accuracy_plot(seed_data: dict[int, pd.DataFrame], output_dir: Path, 
                                     dataset: str, output_filename: str = 'validation_accuracy.png'):
    """Create a validation accuracy plot with all seeds.
    
    Args:
        seed_data: Dict mapping seed number to DataFrame with training logs
        output_dir: Directory to save the plot
        dataset: Dataset name for title
        output_filename: Name of the output file
    """
    if not seed_data:
        print("No data found.")
        return False
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Color palette for accuracy (warm colors - oranges/reds gradient)
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(seed_data)))
    
    # Different markers for each seed
    markers = ['o', 's', '^', 'D', 'v']
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    
    formatter = FuncFormatter(compact_tick_formatter)
    legend_handles = []
    
    max_epoch = 0
    
    # Plot each seed
    for i, (seed, df) in enumerate(sorted(seed_data.items())):
        if 'epoch' not in df.columns or 'val_acc' not in df.columns:
            continue
        
        # Filter out the initial epoch (-1) if present
        df_plot = df[df['epoch'] >= 0].copy()
        
        if df_plot.empty:
            continue
        
        max_epoch = max(max_epoch, df_plot['epoch'].max())
        
        # Plot line
        ax.plot(df_plot['epoch'], df_plot['val_acc'],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=4.0,
                color=colors[i],
                alpha=0.9,
                zorder=2)
        
        # Add markers
        n_points = len(df_plot)
        markevery = max(1, n_points // 8)  # Show ~8 markers
        marker_offset = i % 3
        ax.plot(df_plot['epoch'], df_plot['val_acc'],
                marker=markers[i % len(markers)],
                linestyle='',
                markersize=14,
                color=colors[i],
                markeredgewidth=1.5,
                markeredgecolor='white',
                markevery=(marker_offset * (markevery // 3), markevery),
                zorder=10)
        
        # Legend handle
        legend_handles.append(Line2D([0], [0],
                                     color=colors[i],
                                     linestyle=linestyles[i % len(linestyles)],
                                     linewidth=3.0,
                                     marker=markers[i % len(markers)],
                                     markersize=10,
                                     markeredgewidth=1.5,
                                     markeredgecolor='white',
                                     label=f'Seed {seed}'))
    
    # Labels and grid
    ax.set_xlabel('Epoch', fontsize=30, fontweight='bold')
    ax.set_ylabel('Val Accuracy (%)', fontsize=30, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=2)
    
    # Set axis limits
    ax.set_xlim(-5, max_epoch + 5)
    ax.set_ylim(30, 100)
    
    # Format x-axis with compact formatter
    ax.xaxis.set_major_formatter(formatter)
    
    # Add legend
    legend = ax.legend(handles=legend_handles,
                       loc='lower right',
                       fontsize=22,
                       framealpha=0.95,
                       edgecolor='black',
                       fancybox=False,
                       shadow=False)
    for lh in legend.legend_handles:
        lh.set_linewidth(4.0)
        lh.set_markersize(12)
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Plot validation accuracy over training epochs for all seeds')
    parser.add_argument('--dataset', default='cifar10',
                       help='Dataset name (cifar10 or cifar100, default: cifar10)')
    parser.add_argument('--results-dir', default=None,
                       help='Results directory (default: experiments/results/train_model_and_compare_representations)')
    parser.add_argument('--output-dir', default='../../plots',
                       help='Output directory for plots (default: ../../plots)')
    parser.add_argument('--output-filename', default=None,
                       help='Output filename (default: validation_accuracy_<dataset>.png)')

    args = parser.parse_args()

    # Find results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(__file__).parent.parent / 'results' / 'train_model_and_compare_representations'
    
    results_dir = results_dir.resolve()

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run the train_model_and_compare_representations.py script first to generate results.")
        return

    # Create output directory
    if args.output_dir.startswith('../'):
        output_dir = Path(__file__).parent / args.output_dir
    else:
        output_dir = Path(args.output_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    # Set output filename
    output_filename = args.output_filename or f'validation_accuracy_{args.dataset}.png'

    print(f"Loading data from: {results_dir}")
    print(f"Dataset: {args.dataset}")
    
    # Load data for all seeds
    seed_data = load_all_seeds(results_dir, args.dataset)
    
    if not seed_data:
        print(f"No training data found for dataset: {args.dataset}")
        return
    
    print(f"Found {len(seed_data)} seeds: {sorted(seed_data.keys())}")
    
    for seed, df in sorted(seed_data.items()):
        print(f"  Seed {seed}: {len(df)} epochs, final val_acc: {df['val_acc'].iloc[-1]:.2f}%")
    
    # Create plot
    print(f"\nGenerating validation accuracy plot...")
    success = create_validation_accuracy_plot(seed_data, output_dir, args.dataset, output_filename)
    
    if success:
        print(f"\nPlot saved to: {output_dir}")
        print(f"Generated file: {output_filename}")


if __name__ == "__main__":
    main()
