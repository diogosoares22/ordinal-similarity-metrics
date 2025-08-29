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
from pathlib import Path
import argparse

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def find_csv_files(results_dir: Path) -> list[Path]:
    """Find all CSV files in the benchmark_random results directory."""
    csv_files = list(results_dir.glob("*.csv"))
    return sorted(csv_files)  # Sort alphabetically

def load_and_process_data(csv_path: str) -> pd.DataFrame:
    """Load and process the random benchmark data."""
    df = pd.read_csv(csv_path)
    
    # Extract measure names from column headers
    score_columns = [col for col in df.columns if col.endswith('_score')]
    
    score_columns = [col for col in score_columns if col != 'tsi_score']
    
    # The measure names are already clean from baselines.py
    # Just extract the base name and use as display name
    measure_names = {}
    for col in score_columns:
        measure_name = col.replace('_score', '')
        # Use the measure name directly since they come from BASELINE_MEASURES keys
        measure_names[col] = measure_name
    
    return df, score_columns, measure_names

def create_varying_n_plot(df: pd.DataFrame, score_columns: list, measure_names: dict, output_dir: Path):
    """Create plot for the varying N experiment (fixed D)."""
    n_data = df[df['experiment'] == 'varying_n']
    if n_data.empty:
        print("No varying N data found")
        return
    
    # Calculate averages across runs
    avg_df = n_data.groupby('n_points')[score_columns].mean().reset_index()
    std_df = n_data.groupby('n_points')[score_columns].std().reset_index()
    
    fixed_d = n_data['dim'].iloc[0]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f'Random Case: Similarity Scores vs Dataset Size (Fixed D={fixed_d})', 
                 fontsize=14, fontweight='bold')
    
    # Color palette and markers for better distinguishability
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Plot similarity scores
    for i, col in enumerate(score_columns):
        if col in avg_df.columns:
            # Filter out NaN values
            valid_mask = ~avg_df[col].isna()
            if valid_mask.any():
                x_vals = avg_df.loc[valid_mask, 'n_points']
                y_vals = avg_df.loc[valid_mask, col]
                y_err = std_df.loc[valid_mask, col] if col in std_df.columns else None
                
                ax.errorbar(x_vals, y_vals, yerr=y_err, 
                           marker=markers[i % len(markers)], linewidth=2, markersize=7,
                           color=colors[i % len(colors)], label=measure_names[col],
                           capsize=3, alpha=0.9, markeredgewidth=0.5, markeredgecolor='white')
    
    ax.set_xlabel('Number of Data Points (N)', fontsize=12)
    ax.set_ylabel('Similarity Score', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Set reasonable y-limits for random case (should be low similarity)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'random_benchmark_varying_n.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_varying_d_plot(df: pd.DataFrame, score_columns: list, measure_names: dict, output_dir: Path):
    """Create plot for the varying D experiment (fixed N)."""
    d_data = df[df['experiment'] == 'varying_d']
    if d_data.empty:
        print("No varying D data found")
        return
    
    # Calculate averages across runs
    avg_df = d_data.groupby('dim')[score_columns].mean().reset_index()
    std_df = d_data.groupby('dim')[score_columns].std().reset_index()
    
    fixed_n = d_data['n_points'].iloc[0]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f'Random Case: Similarity Scores vs Dimensionality (Fixed N={fixed_n})', 
                 fontsize=14, fontweight='bold')
    
    # Color palette and markers for better distinguishability
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Plot similarity scores
    for i, col in enumerate(score_columns):
        if col in avg_df.columns:
            # Filter out NaN values
            valid_mask = ~avg_df[col].isna()
            if valid_mask.any():
                x_vals = avg_df.loc[valid_mask, 'dim']
                y_vals = avg_df.loc[valid_mask, col]
                y_err = std_df.loc[valid_mask, col] if col in std_df.columns else None
                
                ax.errorbar(x_vals, y_vals, yerr=y_err,
                           marker=markers[i % len(markers)], linewidth=2, markersize=7,
                           color=colors[i % len(colors)], label=measure_names[col],
                           capsize=3, alpha=0.9, markeredgewidth=0.5, markeredgecolor='white')
    
    ax.set_xlabel('Dimensionality (D)', fontsize=12)
    ax.set_ylabel('Similarity Score', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Set reasonable y-limits for random case (should be low similarity)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'random_benchmark_varying_d.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot random benchmark results')
    parser.add_argument('--output-dir', default='../../plots', 
                       help='Output directory for plots (default: ../../plots)')
    
    args = parser.parse_args()
    
    # Find results directory
    results_dir = Path(__file__).parent.parent / 'results' / 'benchmark_random'
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run the benchmark_random.py script first to generate results.")
        return
    
    # Find CSV files
    csv_files = find_csv_files(results_dir)
    
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        print("Run the benchmark_random.py script first to generate results.")
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
    
    print(f"Creating plots for random benchmark results...")
    print(f"Data shape: {df.shape}")
    print(f"Experiments found: {df['experiment'].unique().tolist()}")
    print(f"Measures found: {list(measure_names.values())}")
    
    # Create plots based on available experiments
    if 'varying_n' in df['experiment'].values:
        print("Creating varying N plot...")
        create_varying_n_plot(df, score_columns, measure_names, output_dir)
    
    if 'varying_d' in df['experiment'].values:
        print("Creating varying D plot...")
        create_varying_d_plot(df, score_columns, measure_names, output_dir)
    
    print(f"\nPlots saved to: {output_dir}")
    print("Generated files:")
    if 'varying_n' in df['experiment'].values:
        print("  - random_benchmark_varying_n.png: Similarity scores vs dataset size")
    if 'varying_d' in df['experiment'].values:
        print("  - random_benchmark_varying_d.png: Similarity scores vs dimensionality")

if __name__ == "__main__":
    main() 