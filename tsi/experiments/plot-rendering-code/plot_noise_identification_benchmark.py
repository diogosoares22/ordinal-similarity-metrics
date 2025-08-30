#!/usr/bin/env python3
"""
Plot script for noise identification benchmark results.
Creates visualizations of similarity measures for noise detection experiment:
- Varying noise level (a) in Y(a) = (1-a) * X + a * Noise

Shows how similarity scores evolve as noise level increases from 0 (perfect similarity) to 1 (pure noise).
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
    """Find all CSV files in the benchmark_noise_identification results directory."""
    csv_files = list(results_dir.glob("*.csv"))
    return sorted(csv_files)  # Sort alphabetically

def load_and_process_data(csv_path: str) -> pd.DataFrame:
    """Load and process the noise identification benchmark data."""
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

def create_noise_identification_plot(df: pd.DataFrame, score_columns: list, measure_names: dict, output_dir: Path):
    """Create plot for the noise identification experiment (varying noise level)."""
    noise_data = df[df['experiment'] == 'varying_noise_uniform']
    if noise_data.empty:
        print("No noise identification data found")
        return
    
    # Calculate averages across runs
    avg_df = noise_data.groupby('noise_level')[score_columns].mean().reset_index()
    std_df = noise_data.groupby('noise_level')[score_columns].std().reset_index()
    
    fixed_n = noise_data['n_points'].iloc[0]
    fixed_dim = noise_data['dim'].iloc[0]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(f'Noise Identification: Similarity Scores vs Noise Level\nY(a) = (1-a)·X + a·Noise (N={fixed_n}, D={fixed_dim})', 
                 fontsize=14, fontweight='bold')
    
    # Use color scheme from other benchmark plots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Plot similarity scores
    for i, col in enumerate(score_columns):
        if col in avg_df.columns:
            # Filter out NaN values
            valid_mask = ~avg_df[col].isna()
            if valid_mask.any():
                x_vals = avg_df.loc[valid_mask, 'noise_level']
                y_vals = avg_df.loc[valid_mask, col]
                y_err = std_df.loc[valid_mask, col] if col in std_df.columns else None
                
                # Use thicker line for TSI
                linewidth = 3 if col == 'TSI_score' else 2
                markersize = 8 if col == 'TSI_score' else 7
                alpha = 1.0 if col == 'TSI_score' else 0.9
                
                ax.errorbar(x_vals, y_vals, yerr=y_err,
                           marker=markers[i % len(markers)], linewidth=linewidth, markersize=markersize,
                           color=colors[i % len(colors)], label=measure_names[col],
                           capsize=3, alpha=alpha, markeredgewidth=0.5, markeredgecolor='white')
    
    ax.set_xlabel('Noise Level (a)', fontsize=12)
    ax.set_ylabel('Similarity Score', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set reasonable y-limits
    ax.set_ylim(-0.05, 1.05)
    
    # Add vertical lines and annotations for key noise levels
    ax.axvline(x=0.0, color='green', linestyle='--', alpha=0.7, linewidth=1)
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    # Add text annotations
    ax.text(0.02, 0.95, 'Perfect\nSimilarity\n(Y = X)', transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.text(0.98, 0.05, 'Pure Noise\n(Y = Noise)', transform=ax.transAxes, 
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # Add legend after all plot elements
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'noise_identification_benchmark.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_noise_sensitivity_plot(df: pd.DataFrame, score_columns: list, measure_names: dict, output_dir: Path):
    """Create a plot showing noise sensitivity (derivative of similarity w.r.t. noise level)."""
    noise_data = df[df['experiment'] == 'varying_noise_uniform']
    if noise_data.empty:
        print("No noise identification data found for sensitivity plot")
        return
    
    # Calculate averages across runs
    avg_df = noise_data.groupby('noise_level')[score_columns].mean().reset_index()
    
    fixed_n = noise_data['n_points'].iloc[0]
    fixed_dim = noise_data['dim'].iloc[0]
    
    # Calculate sensitivity (negative derivative approximation)
    sensitivity_data = {}
    for col in score_columns:
        if col in avg_df.columns and not avg_df[col].isna().all():
            # Calculate finite differences (negative for sensitivity)
            noise_levels = avg_df['noise_level'].values
            scores = avg_df[col].values
            
            # Remove NaN values
            valid_mask = ~np.isnan(scores)
            if np.sum(valid_mask) > 1:
                noise_levels_clean = noise_levels[valid_mask]
                scores_clean = scores[valid_mask]
                
                # Calculate sensitivity as -d(score)/d(noise_level)
                sensitivity = -np.gradient(scores_clean, noise_levels_clean)
                sensitivity_data[col] = {
                    'noise_levels': noise_levels_clean,
                    'sensitivity': sensitivity
                }
    
    if not sensitivity_data:
        print("No valid data for sensitivity plot")
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(f'Noise Sensitivity: Rate of Similarity Decrease\n-d(Similarity)/d(Noise Level) (N={fixed_n}, D={fixed_dim})', 
                 fontsize=14, fontweight='bold')
    
    # Use color scheme from other benchmark plots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Plot sensitivity
    for i, col in enumerate(score_columns):
        if col in sensitivity_data:
            data = sensitivity_data[col]
            x_vals = data['noise_levels']
            y_vals = data['sensitivity']
            
            # Use thicker line for TSI
            linewidth = 3 if col == 'TSI_score' else 2
            markersize = 8 if col == 'TSI_score' else 7
            alpha = 1.0 if col == 'TSI_score' else 0.9
            
            ax.plot(x_vals, y_vals,
                   marker=markers[i % len(markers)], linewidth=linewidth, markersize=markersize,
                   color=colors[i % len(colors)], label=measure_names[col],
                   alpha=alpha, markeredgewidth=0.5, markeredgecolor='white')
    
    ax.set_xlabel('Noise Level (a)', fontsize=12)
    ax.set_ylabel('Noise Sensitivity (-dS/da)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Add legend after all plot elements
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'noise_sensitivity_benchmark.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_combined_plot(df: pd.DataFrame, score_columns: list, measure_names: dict, output_dir: Path):
    """Create a combined plot with both similarity scores and sensitivity."""
    noise_data = df[df['experiment'] == 'varying_noise_uniform']
    if noise_data.empty:
        print("No noise identification data found for combined plot")
        return
    
    # Calculate averages across runs
    avg_df = noise_data.groupby('noise_level')[score_columns].mean().reset_index()
    
    fixed_n = noise_data['n_points'].iloc[0]
    fixed_dim = noise_data['dim'].iloc[0]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'Noise Identification Analysis: Y(a) = (1-a)·X + a·Noise (N={fixed_n}, D={fixed_dim})', 
                 fontsize=14, fontweight='bold')
    
    # Use color scheme from other benchmark plots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Plot 1: Similarity scores
    for i, col in enumerate(score_columns):
        if col in avg_df.columns:
            # Filter out NaN values
            valid_mask = ~avg_df[col].isna()
            if valid_mask.any():
                x_vals = avg_df.loc[valid_mask, 'noise_level']
                y_vals = avg_df.loc[valid_mask, col]
                
                # Use thicker line for TSI
                linewidth = 3 if col == 'TSI_score' else 2
                markersize = 8 if col == 'TSI_score' else 7
                alpha = 1.0 if col == 'TSI_score' else 0.9
                
                ax1.plot(x_vals, y_vals,
                        marker=markers[i % len(markers)], linewidth=linewidth, markersize=markersize,
                        color=colors[i % len(colors)], label=measure_names[col],
                        alpha=alpha, markeredgewidth=0.5, markeredgecolor='white')
    
    ax1.set_xlabel('Noise Level (a)', fontsize=12)
    ax1.set_ylabel('Similarity Score', fontsize=12)
    ax1.set_title('Similarity Scores vs Noise Level', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Plot 2: Sensitivity (derivative)
    sensitivity_data = {}
    for col in score_columns:
        if col in avg_df.columns and not avg_df[col].isna().all():
            noise_levels = avg_df['noise_level'].values
            scores = avg_df[col].values
            
            valid_mask = ~np.isnan(scores)
            if np.sum(valid_mask) > 1:
                noise_levels_clean = noise_levels[valid_mask]
                scores_clean = scores[valid_mask]
                sensitivity = -np.gradient(scores_clean, noise_levels_clean)
                sensitivity_data[col] = {
                    'noise_levels': noise_levels_clean,
                    'sensitivity': sensitivity
                }
    
    for i, col in enumerate(score_columns):
        if col in sensitivity_data:
            data = sensitivity_data[col]
            x_vals = data['noise_levels']
            y_vals = data['sensitivity']
            
            linewidth = 3 if col == 'TSI_score' else 2
            markersize = 8 if col == 'TSI_score' else 7
            alpha = 1.0 if col == 'TSI_score' else 0.9
            
            ax2.plot(x_vals, y_vals,
                    marker=markers[i % len(markers)], linewidth=linewidth, markersize=markersize,
                    color=colors[i % len(colors)], label=measure_names[col],
                    alpha=alpha, markeredgewidth=0.5, markeredgecolor='white')
    
    ax2.set_xlabel('Noise Level (a)', fontsize=12)
    ax2.set_ylabel('Noise Sensitivity (-dS/da)', fontsize=12)
    ax2.set_title('Noise Sensitivity (Rate of Similarity Decrease)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'noise_identification_combined.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot noise identification benchmark results')
    parser.add_argument('--output-dir', default='../../plots', 
                       help='Output directory for plots (default: ../../plots)')
    parser.add_argument('--plot-type', choices=['similarity', 'sensitivity', 'combined', 'all'], default='all',
                       help='Type of plot to generate (default: all)')
    
    args = parser.parse_args()
    
    # Find results directory
    results_dir = Path(__file__).parent.parent / 'results' / 'benchmark_noise_identification'
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run the benchmark_noise_identification.py script first to generate results.")
        return
    
    # Find CSV files
    csv_files = find_csv_files(results_dir)
    
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        print("Run the benchmark_noise_identification.py script first to generate results.")
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
    
    print(f"Creating plots for noise identification benchmark results...")
    print(f"Data shape: {df.shape}")
    print(f"Experiments found: {df['experiment'].unique().tolist()}")
    print(f"Measures found: {list(measure_names.values())}")
    
    # Create plots based on selection
    generated_files = []
    
    if args.plot_type in ['similarity', 'all']:
        print("Creating similarity scores plot...")
        create_noise_identification_plot(df, score_columns, measure_names, output_dir)
        generated_files.append("  - noise_identification_benchmark.png: Similarity scores vs noise level")
    
    if args.plot_type in ['sensitivity', 'all']:
        print("Creating noise sensitivity plot...")
        create_noise_sensitivity_plot(df, score_columns, measure_names, output_dir)
        generated_files.append("  - noise_sensitivity_benchmark.png: Noise sensitivity analysis")
    
    if args.plot_type in ['combined', 'all']:
        print("Creating combined plot...")
        create_combined_plot(df, score_columns, measure_names, output_dir)
        generated_files.append("  - noise_identification_combined.png: Combined similarity and sensitivity plots")
    
    print(f"\nPlots saved to: {output_dir}")
    print("Generated files:")
    for file_desc in generated_files:
        print(file_desc)

if __name__ == "__main__":
    main() 