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

def load_and_process_data(csv_path: str) -> pd.DataFrame:
    """Load and process the benchmark data."""
    df = pd.read_csv(csv_path)
    
    # Extract measure names from column headers
    time_columns = [col for col in df.columns if col.endswith('_time')]
    measure_names = []
    
    for col in time_columns:
        # Extract measure name from column header
        measure_name = col.replace('_time', '')
        # Clean up the measure name for display
        if 'cka' in measure_name:
            display_name = 'CKA'
        elif 'cknna' in measure_name:
            display_name = 'CKNNA'
        elif 'mutual_knn' in measure_name:
            display_name = 'Mutual KNN'
        elif 'svcca' in measure_name:
            if 'angular' in measure_name:
                display_name = 'SVCCA (Angular)'
            elif 'euclidean' in measure_name:
                display_name = 'SVCCA (Euclidean)'
            else:
                display_name = 'SVCCA'
        else:
            display_name = measure_name.replace('measure/platonic/', '').replace('_', ' ').title()
        
        measure_names.append(display_name)
    
    return df, time_columns, measure_names

def create_latency_plot(df: pd.DataFrame, time_columns: list, measure_names: list, output_path: str):
    """Create a comprehensive latency comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Linear scale for smaller values
    for i, (col, name) in enumerate(zip(time_columns, measure_names)):
        ax1.plot(df['n_points'], df[col], marker='o', linewidth=2, markersize=6, label=name)
    
    ax1.set_xlabel('Number of Data Points', fontsize=12)
    ax1.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax1.set_title('Latency Comparison (Linear Scale)', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log scale for better visualization of larger differences
    for i, (col, name) in enumerate(zip(time_columns, measure_names)):
        ax2.plot(df['n_points'], df[col], marker='o', linewidth=2, markersize=6, label=name)
    
    ax2.set_xlabel('Number of Data Points', fontsize=12)
    ax2.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax2.set_title('Latency Comparison (Log Scale)', fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load data
    csv_path = Path(__file__).parent.parent / 'results' / 'latency_benchmark_xticks30_factor200_initial100.csv'
    df, time_columns, measure_names = load_and_process_data(str(csv_path))
    
    # Create output directory
    output_dir = Path(__file__).parent.parent.parent / 'plots'
    output_dir.mkdir(exist_ok=True)
    
    # Create different types of plots
    create_latency_plot(df, time_columns, measure_names, 
                       output_dir / 'latency_comparison.png')
    
    print(f"Plots saved to: {output_dir}")
    print("Generated files:")
    print("  - latency_comparison.png: Line plots comparing all measures")
    
if __name__ == "__main__":
    main()
