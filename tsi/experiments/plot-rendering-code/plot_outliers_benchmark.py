#!/usr/bin/env python3
"""
Plot script for outliers benchmark results.
Creates visualizations of similarity measures for two experiments:
1. Varying sigma (standard deviations) with PC direction outliers
2. Varying sigma (standard deviations) with random direction outliers

Shows how outliers affect similarity scores between X and X' where X' has k outliers.
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
    """Find all CSV files in the benchmark_outliers results directory."""
    csv_files = list(results_dir.glob("*.csv"))
    return sorted(csv_files)  # Sort alphabetically

def load_and_process_data(csv_path: str) -> pd.DataFrame:
    """Load and process the outliers benchmark data."""
    df = pd.read_csv(csv_path)
    
    # Extract measure names from column headers
    score_columns = [col for col in df.columns if col.endswith('_score')]
    
    # Remove TSI from score_columns since we want to focus on baselines vs TSI
    score_columns = [col for col in score_columns if col != 'TSI_score']
    
    # The measure names are already clean from baselines.py
    # Just extract the base name and use as display name
    measure_names = {}
    for col in score_columns:
        measure_name = col.replace('_score', '')
        # Use the measure name directly since they come from BASELINE_MEASURES keys
        measure_names[col] = measure_name
    
    # Add TSI if it exists
    if 'TSI_score' in df.columns:
        score_columns.insert(0, 'TSI_score')  # Put TSI first
        measure_names['TSI_score'] = 'TSI'
    
    return df, score_columns, measure_names

def create_pc_direction_plot(df: pd.DataFrame, score_columns: list, measure_names: dict, output_dir: Path):
    """Create plot for the PC direction experiment (varying sigma with PC direction)."""
    pc_data = df[df['experiment'] == 'varying_sigma_pc']
    if pc_data.empty:
        print("No PC direction data found")
        return
    
    # Calculate averages across runs
    avg_df = pc_data.groupby('sigma')[score_columns].mean().reset_index()
    
    # Rename sigma column to alpha for display
    avg_df = avg_df.rename(columns={'sigma': 'alpha'})
    
    fixed_n = pc_data['n_points'].iloc[0]
    k_outliers = pc_data['k_outliers'].iloc[0]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f'Outliers Impact: PC Direction - Similarity Scores vs Outlier Strength (N={fixed_n}, k={k_outliers})', 
                 fontsize=14, fontweight='bold')
    
    # Use color scheme from random benchmark
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Plot similarity scores
    for i, col in enumerate(score_columns):
        if col in avg_df.columns:
            # Filter out NaN values
            valid_mask = ~avg_df[col].isna()
            if valid_mask.any():
                x_vals = avg_df.loc[valid_mask, 'alpha']
                y_vals = avg_df.loc[valid_mask, col]
                
                # Use thicker line for TSI
                linewidth = 3 if col == 'TSI_score' else 2
                markersize = 8 if col == 'TSI_score' else 7
                alpha = 1.0 if col == 'TSI_score' else 0.9
                
                ax.plot(x_vals, y_vals, 
                       marker=markers[i % len(markers)], linewidth=linewidth, markersize=markersize,
                       color=colors[i % len(colors)], label=measure_names[col],
                       alpha=alpha, markeredgewidth=0.5, markeredgecolor='white')
    
    ax.set_xlabel('Outlier Strength (α)', fontsize=12)
    ax.set_ylabel('Similarity Score', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set reasonable y-limits
    ax.set_ylim(-0.05, 1.05)
    
    # Add legend after all plot elements
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'outliers_benchmark_pc_direction.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_random_direction_plot(df: pd.DataFrame, score_columns: list, measure_names: dict, output_dir: Path):
    """Create plot for the random direction experiment (varying sigma with random direction)."""
    random_data = df[df['experiment'] == 'varying_sigma_random']
    if random_data.empty:
        print("No random direction data found")
        return
    
    # Calculate averages across runs
    avg_df = random_data.groupby('sigma')[score_columns].mean().reset_index()
    
    # Rename sigma column to alpha for display
    avg_df = avg_df.rename(columns={'sigma': 'alpha'})
    
    fixed_n = random_data['n_points'].iloc[0]
    k_outliers = random_data['k_outliers'].iloc[0]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f'Outliers Impact: Random Direction - Similarity Scores vs Outlier Strength (N={fixed_n}, k={k_outliers})', 
                 fontsize=14, fontweight='bold')
    
    # Use color scheme from random benchmark
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Plot similarity scores
    for i, col in enumerate(score_columns):
        if col in avg_df.columns:
            # Filter out NaN values
            valid_mask = ~avg_df[col].isna()
            if valid_mask.any():
                x_vals = avg_df.loc[valid_mask, 'alpha']
                y_vals = avg_df.loc[valid_mask, col]
                
                # Use thicker line for TSI
                linewidth = 3 if col == 'TSI_score' else 2
                markersize = 8 if col == 'TSI_score' else 7
                alpha = 1.0 if col == 'TSI_score' else 0.9
                
                ax.plot(x_vals, y_vals, 
                       marker=markers[i % len(markers)], linewidth=linewidth, markersize=markersize,
                       color=colors[i % len(colors)], label=measure_names[col],
                       alpha=alpha, markeredgewidth=0.5, markeredgecolor='white')
    
    ax.set_xlabel('Outlier Strength (α)', fontsize=12)
    ax.set_ylabel('Similarity Score', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set reasonable y-limits
    ax.set_ylim(-0.05, 1.05)
    
    # Add legend after all plot elements
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'outliers_benchmark_random_direction.png', dpi=300, bbox_inches='tight')
    plt.close()



def main():
    parser = argparse.ArgumentParser(description='Plot outliers benchmark results')
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
    print(f"Experiments found: {df['experiment'].unique().tolist()}")
    print(f"Measures found: {list(measure_names.values())}")
    
    # Create plots based on available experiments
    if 'varying_sigma_pc' in df['experiment'].values:
        print("Creating PC direction plot...")
        create_pc_direction_plot(df, score_columns, measure_names, output_dir)
    
    if 'varying_sigma_random' in df['experiment'].values:
        print("Creating random direction plot...")
        create_random_direction_plot(df, score_columns, measure_names, output_dir)
    
    print(f"\nPlots saved to: {output_dir}")
    print("Generated files:")
    if 'varying_sigma_pc' in df['experiment'].values:
        print("  - outliers_benchmark_pc_direction.png: PC direction outliers vs outlier strength")
    if 'varying_sigma_random' in df['experiment'].values:
        print("  - outliers_benchmark_random_direction.png: Random direction outliers vs outlier strength")

if __name__ == "__main__":
    main() 