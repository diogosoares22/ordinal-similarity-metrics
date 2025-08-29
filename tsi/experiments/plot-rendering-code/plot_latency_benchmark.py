#!/usr/bin/env python3
"""
Plot script for latency benchmark results.
Creates a comprehensive visualization of computation times for different measures.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def find_csv_files(results_dir: Path) -> list[Path]:
    """Find all CSV files in the results directory."""
    csv_files = list(results_dir.glob("*.csv"))
    return sorted(csv_files)  # Sort alphabetically

def load_and_process_data(csv_path: str) -> pd.DataFrame:
    """Load and process the benchmark data."""
    df = pd.read_csv(csv_path)
    
    # Extract measure names from column headers
    time_columns = [col for col in df.columns if col.endswith('_time')]
    
    # The measure names are already clean from baselines.py
    # Just extract the base name and use as display name
    measure_names = {}
    for col in time_columns:
        measure_name = col.replace('_time', '')
        # Use the measure name directly since they come from BASELINE_MEASURES keys
        measure_names[col] = measure_name
    
    return df, time_columns, measure_names

def create_latency_plot(df: pd.DataFrame, time_columns: list, measure_names: dict, output_path: str):
    """Create a comprehensive latency comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color palette and markers for better distinguishability
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Plot 1: Linear scale for smaller values
    for i, col in enumerate(time_columns):
        if col in df.columns:
            # Filter out NaN values
            valid_mask = ~df[col].isna()
            if valid_mask.any():
                x_vals = df.loc[valid_mask, 'n_points']
                y_vals = df.loc[valid_mask, col]
                
                ax1.plot(x_vals, y_vals, marker=markers[i % len(markers)], 
                        linewidth=2, markersize=6, color=colors[i % len(colors)], 
                        label=measure_names[col], alpha=0.9, markeredgewidth=0.5, markeredgecolor='white')
    
    ax1.set_xlabel('Number of Data Points', fontsize=12)
    ax1.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax1.set_title('Latency Comparison (Linear Scale)', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log scale for better visualization of larger differences
    for i, col in enumerate(time_columns):
        if col in df.columns:
            # Filter out NaN values and zero values for log scale
            valid_mask = (~df[col].isna()) & (df[col] > 0)
            if valid_mask.any():
                x_vals = df.loc[valid_mask, 'n_points']
                y_vals = df.loc[valid_mask, col]
                
                ax2.plot(x_vals, y_vals, marker=markers[i % len(markers)], 
                        linewidth=2, markersize=6, color=colors[i % len(colors)], 
                        label=measure_names[col], alpha=0.9, markeredgewidth=0.5, markeredgecolor='white')
    
    ax2.set_xlabel('Number of Data Points', fontsize=12)
    ax2.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax2.set_title('Latency Comparison (Log Scale)', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Find results directory
    results_dir = Path(__file__).parent.parent / 'results/benchmark_latency'
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run the latency benchmark script first to generate results.")
        return
    
    # Find CSV files
    csv_files = find_csv_files(results_dir)
    
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        print("Run the latency benchmark script first to generate results.")
        return
    
    # Filter for latency benchmark files (look for files with "latency" in the name)
    latency_files = [f for f in csv_files if 'latency' in f.name.lower()]
    
    if latency_files:
        csv_path = latency_files[0]  # Use first latency file found
        print(f"Using latency CSV file: {csv_path.name}")
    else:
        # Fallback to first CSV file if no latency-specific files found
        csv_path = csv_files[0]
        print(f"No latency-specific CSV found, using: {csv_path.name}")
    
    # Load and process data
    df, time_columns, measure_names = load_and_process_data(str(csv_path))
    
    # Create output directory
    output_dir = Path(__file__).parent.parent.parent / 'plots'
    output_dir.mkdir(exist_ok=True)
    
    print(f"Creating latency comparison plot...")
    print(f"Data shape: {df.shape}")
    print(f"Measures found: {list(measure_names.values())}")
    
    # Create different types of plots
    create_latency_plot(df, time_columns, measure_names, 
                       output_dir / 'latency_comparison.png')
    
    print(f"Plots saved to: {output_dir}")
    print("Generated files:")
    print("  - latency_comparison.png: Line plots comparing all measures")
    
if __name__ == "__main__":
    main()
