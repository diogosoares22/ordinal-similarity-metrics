#!/usr/bin/env python3
"""
Benchmark script for measuring latency of baseline measures vs TSI.
Takes command line arguments for number of xticks, additive factor, and initial datapoints.
"""

import argparse
import time
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tsi.baselines import run_baseline_measures
from tsi.tsi import EfficientTSI, RepresentationPair


def generate_data(n_points: int, dim: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Generate random data for benchmarking."""
    X = np.random.rand(n_points, dim)
    Y = np.random.rand(n_points, dim)
    return X, Y


def benchmark_tsi(X: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
    """Benchmark TSI computation time."""
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    efficient_tsi = EfficientTSI(euclidean=True, memory_efficient=False)
    start_time = time.time()
    score = efficient_tsi(representations)
    end_time = time.time()
    
    return score, end_time - start_time


def benchmark_baselines(X: np.ndarray, Y: np.ndarray) -> dict:
    """Benchmark baseline measures computation time."""
    return run_baseline_measures(X, Y, time_monitor=True)


def main():
    parser = argparse.ArgumentParser(description='Benchmark latency of baseline measures vs TSI')
    parser.add_argument('--xticks', type=int, default=10, required=False, 
                       help='Number of data points to test (x-axis ticks)')
    parser.add_argument('--factor', type=float, default=500, required=False,
                       help='Additives factor to increase data points')
    parser.add_argument('--initial', type=int, default=1000, required=False,
                       help='Initial number of data points')
    parser.add_argument('--with-tsi', action='store_true', default=True,
                       help='Whether to include TSI in the benchmark')
    
    args = parser.parse_args()
    
    data_sizes = [int(args.initial + (args.factor * i)) for i in range(args.xticks)]
    
    max_size = min(max(data_sizes), 50000)
    data_sizes = [size for size in data_sizes if size <= max_size]
    
    print(f"Benchmarking with data sizes: {data_sizes}")
    
    results = []
    
    for n_points in data_sizes:
        print(f"Testing with {n_points} data points...")
        
        X, Y = generate_data(n_points)
        
        result_row = {
            'n_points': n_points,
        }
        
        if args.with_tsi:
            tsi_score, tsi_time = benchmark_tsi(X, Y)
            result_row['tsi_score'] = tsi_score
            result_row['tsi_time'] = tsi_time
        
        baseline_results = benchmark_baselines(X, Y)
        
        for measure_name, (score, time_taken) in baseline_results.items():
            result_row[f'{measure_name}_score'] = score
            result_row[f'{measure_name}_time'] = time_taken
        
        results.append(result_row)
        
        if args.with_tsi:
            print(f"  TSI time: {tsi_time:.4f}s")
        print(f"  Baseline times: {[f'{time_taken:.4f}s' for _, time_taken in baseline_results.values()]}")
    
    df = pd.DataFrame(results)
    
    results_dir = Path(__file__).parent.parent / 'results/benchmark_latency'
    results_dir.mkdir(exist_ok=True)
    
    filename = f"latency_benchmark_xticks{args.xticks}_factor{args.factor}_initial{args.initial}.csv"
    filepath = results_dir / filename
    
    df.to_csv(filepath, index=False)
    print(f"\nResults saved to: {filepath}")
    
    print("\nBenchmark Summary:")
    print(f"Data sizes tested: {data_sizes}")
    print(f"Total measurements: {len(results)}")
    print(f"Measures compared: TSI vs {len(baseline_results)} baseline measures")


if __name__ == "__main__":
    main()

