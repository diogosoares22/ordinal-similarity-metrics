#!/usr/bin/env python3
"""
Plot script for CLIP visual vs textual representation comparison results on ImageNet.
Creates a grouped bar chart showing similarity scores across different metrics
for small, medium, and large CLIP models comparing image and text representations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from pathlib import Path
import argparse

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


# Metrics to display and their display names
METRIC_GROUPS = {
    'Ordinal': ['C-TSI', 'C-QSI', 'C-TSI-CosSim', 'C-QSI-CosSim'],
    'Baseline': ['C-CKA', 'B-CKNNA', 'B-MutualNN'],
}

METRIC_DISPLAY_NAMES = {
    'B-TSI': 'B-TSI',
    'B-QSI': 'B-QSI',
    'C-TSI': 'TSI \n(Euc)',
    'C-QSI': 'QSI \n(Euc)',
    'C-TSI-CosSim': 'TSI \n(Cos)',
    'C-QSI-CosSim': 'QSI \n(Cos)',
    'B-CKA': 'B-CKA',
    'C-CKA': 'CKA',
    'B-CKNNA': 'CKNNA',
    'B-MutualNN': 'MutualNN',
    'B-SVCCA': 'SVCCA',
    'B-PWCCA': 'PWCCA',
}

MODEL_DISPLAY_NAMES = {
    'small': 'Small',
    'medium': 'Medium',
    'large': 'Large',
}


def find_csv_files(results_dir: Path) -> list[Path]:
    """Find all CSV files in the results directory."""
    csv_files = list(results_dir.glob("imagenet_*_visual_textual_comparison*.csv"))
    return sorted(csv_files, key=lambda x: x.stat().st_mtime, reverse=True)


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and process the comparison data."""
    df = pd.read_csv(csv_path)
    return df


def get_available_metrics(df: pd.DataFrame) -> list[str]:
    """Get list of available metrics in the dataframe."""
    # Reserved columns that are not metrics
    reserved = {'model_size', 'split', 'run', 'seed', 'n_samples', 'image_dim', 
                'text_dim', 'batch_size', 'no_batches'}
    
    metrics = []
    for col in df.columns:
        if col not in reserved and not df[col].isna().all():
            metrics.append(col)
    
    return metrics


def create_grouped_bar_plot(df: pd.DataFrame, output_dir: Path, output_name: str = None):
    """
    Create a grouped bar chart showing similarity scores per model size.
    
    Args:
        df: DataFrame with similarity scores
        output_dir: Directory to save the plot
        output_name: Optional custom output filename
    """
    # Get available metrics
    available_metrics = get_available_metrics(df)
    
    # Order metrics: TSI/QSI variants first, then baselines
    ordered_metrics = []
    preferred_order = ['C-TSI', 'C-QSI', 'C-TSI-CosSim', 'C-QSI-CosSim', 
                       'C-CKA', 'B-CKNNA', 'B-MutualNN', 'B-SVCCA', 'B-PWCCA']
    for m in preferred_order:
        if m in available_metrics:
            ordered_metrics.append(m)
    
    
    metrics = ordered_metrics
    model_sizes = ['small', 'medium', 'large']
    
    # Filter to only existing model sizes
    model_sizes = [m for m in model_sizes if m in df['model_size'].values]
    
    # Compute mean and std per model size
    mean_data = df.groupby('model_size')[metrics].mean()
    std_data = df.groupby('model_size')[metrics].std()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Bar positioning
    n_metrics = len(metrics)
    n_models = len(model_sizes)
    bar_width = 0.25
    x = np.arange(n_metrics)
    
    # Colors for each model size
    colors = {
        'small': '#1b7837',   # Green
        'medium': '#2166ac',  # Blue
        'large': '#b2182b',   # Red
    }
    
    # Plot bars for each model size
    for i, model in enumerate(model_sizes):
        if model not in mean_data.index:
            continue
        
        offset = (i - (n_models - 1) / 2) * bar_width
        values = mean_data.loc[model, metrics].values
        errors = std_data.loc[model, metrics].values if len(df[df['model_size'] == model]) > 1 else None
        
        # Handle NaN values
        values = np.where(np.isnan(values), 0, values)
        if errors is not None:
            errors = np.where(np.isnan(errors), 0, errors)
        
        bars = ax.bar(x + offset, values, bar_width, 
                      label=MODEL_DISPLAY_NAMES.get(model, model),
                      color=colors.get(model, f'C{i}'),
                      edgecolor='white',
                      linewidth=1.5,
                      alpha=0.9,
                      zorder=3)
        
        # Add error bars if we have multiple runs
        if errors is not None and np.any(errors > 0):
            ax.errorbar(x + offset, values, yerr=errors,
                       fmt='none', color='black', capsize=4, capthick=2,
                       linewidth=2, zorder=4)
    
    # Customize axes
    ax.set_xlabel('Similarity Metric', fontsize=30, fontweight='bold')
    ax.set_ylabel('Similarity Score', fontsize=30, fontweight='bold')
    
    # X-axis labels
    metric_labels = [METRIC_DISPLAY_NAMES.get(m, m) for m in metrics]
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=30, ha='center', fontsize=28)
    
    # Y-axis limits
    ax.set_ylim(0, 0.7)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=2, axis='y', zorder=0)
    ax.set_axisbelow(True)
    
    # Legend
    legend = ax.legend(loc='upper right', fontsize=26, framealpha=0.95, 
                       edgecolor='black', fancybox=False, title='CLIP Model',
                       title_fontsize=26)
    
    plt.tight_layout()
    
    # Save
    if output_name is None:
        output_name = 'imagenet_visual_textual_comparison.png'
    
    output_path = output_dir / output_name
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Generated: {output_name}")
    return output_path


def create_heatmap_plot(df: pd.DataFrame, output_dir: Path):
    """
    Create a heatmap showing similarity scores with metrics as rows and models as columns.
    
    Args:
        df: DataFrame with similarity scores
        output_dir: Directory to save the plot
    """
    # Get available metrics
    available_metrics = get_available_metrics(df)
    
    # Order metrics
    ordered_metrics = []
    preferred_order = ['C-TSI', 'C-QSI', 'C-TSI-CosSim', 'C-QSI-CosSim', 
                       'C-CKA', 'B-CKNNA', 'B-MutualNN']
    for m in preferred_order:
        if m in available_metrics:
            ordered_metrics.append(m)
    
    metrics = ordered_metrics
    model_sizes = ['small', 'medium', 'large']
    model_sizes = [m for m in model_sizes if m in df['model_size'].values]
    
    # Create pivot table
    mean_data = df.groupby('model_size')[metrics].mean()
    
    # Reorder
    mean_data = mean_data.reindex(model_sizes)
    mean_data = mean_data[metrics].T
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create heatmap
    im = ax.imshow(mean_data.values, cmap='RdYlGn', aspect='auto', 
                   vmin=0, vmax=0.7)
    
    # Customize axes
    ax.set_xticks(np.arange(len(model_sizes)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in model_sizes], fontsize=26)
    ax.set_yticklabels([METRIC_DISPLAY_NAMES.get(m, m) for m in metrics], fontsize=26)
    
    # Add value annotations
    for i in range(len(metrics)):
        for j in range(len(model_sizes)):
            value = mean_data.values[i, j]
            if not np.isnan(value):
                text_color = 'white' if value > 0.4 else 'black'
                ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                       color=text_color, fontsize=22, fontweight='bold')
    
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Similarity Score', fontsize=26, fontweight='bold')
    cbar.ax.tick_params(labelsize=22)
    
    ax.set_xlabel('CLIP Model Size', fontsize=30, fontweight='bold')
    ax.set_ylabel('Similarity Metric', fontsize=30, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'imagenet_visual_textual_comparison_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Generated: imagenet_visual_textual_comparison_heatmap.png")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Plot CLIP visual vs textual representation comparison results (ImageNet)'
    )
    parser.add_argument(
        '--results-dir', 
        type=str,
        default=None,
        help='Results directory (default: experiments/results/clip_visual_textual_comparison)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str,
        default='../../plots',
        help='Output directory for plots (default: ../../plots)'
    )
    parser.add_argument(
        '--csv-file',
        type=str,
        default=None,
        help='Specific CSV file to use (default: most recent in results dir)'
    )
    parser.add_argument(
        '--heatmap',
        action='store_true',
        help='Also generate a heatmap visualization'
    )
    
    args = parser.parse_args()
    
    # Find results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(__file__).parent.parent / 'results' / 'clip_visual_textual_comparison'
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run the compare_visual_textual_representations_clip.py script first to generate results.")
        return
    
    # Find CSV file
    if args.csv_file:
        csv_path = Path(args.csv_file)
        if not csv_path.exists():
            csv_path = results_dir / args.csv_file
    else:
        csv_files = find_csv_files(results_dir)
        if not csv_files:
            print(f"No CSV files found in {results_dir}")
            print("Run the compare_visual_textual_representations_clip.py script first to generate results.")
            return
        csv_path = csv_files[0]  # Most recent
    
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        return
    
    print(f"Using CSV file: {csv_path.name}")
    
    # Load data
    df = load_data(csv_path)
    
    # Create output directory
    if args.output_dir.startswith('../'):
        output_dir = Path(__file__).parent / args.output_dir
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Creating plots for CLIP visual vs textual comparison (ImageNet)...")
    print(f"Data shape: {df.shape}")
    print(f"Model sizes: {df['model_size'].unique().tolist()}")
    if 'split' in df.columns:
        print(f"Split: {df['split'].unique().tolist()}")
    print(f"Runs per model: {df.groupby('model_size').size().to_dict()}")
    
    available_metrics = get_available_metrics(df)
    print(f"Available metrics: {available_metrics}")
    
    # Create main bar plot
    print("\nGenerating grouped bar plot...")
    create_grouped_bar_plot(df, output_dir)
    
    # Optionally create heatmap
    if args.heatmap:
        print("Generating heatmap...")
        create_heatmap_plot(df, output_dir)
    
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
