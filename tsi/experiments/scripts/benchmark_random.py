#!/usr/bin/env python3
"""
Benchmark script for random data similarity measures.
Tests baseline measures on random case: X vs Y (independent random data).
Includes two experiments:
1. Varying N (n_min to n_max) with fixed D=500
2. Varying D (50 to 500) with fixed N=2000
"""

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tsi.baselines import run_baseline_measures
from tsi.tsi import EfficientTSI, RepresentationPair


def generate_random_data(n_points: int, dim: int, seed: int | None):
    """Generate two independent random datasets X and Y."""
    rng = np.random.default_rng(seed)
    # Generate data in range [0.1, 1.0]
    X = rng.random((n_points, dim))
    
    # Use different seed for Y to ensure independence
    rng_y = np.random.default_rng(seed + 1000 if seed is not None else None)
    Y = rng_y.random((n_points, dim))
    
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


def run_n_experiment(args):
    """Run experiment varying N with fixed D."""
    print(f"=== EXPERIMENT 1: Varying N (fixed D={args.fixed_d}) ===")
    
    n_values = list(range(args.n_points_min, args.n_points_max + 1, args.n_points_step))
    fixed_dim = args.fixed_d
    
    print(f"N values: {n_values}")
    print(f"Fixed dimensionality: {fixed_dim}")
    print(f"Runs per N: {args.runs}")
    print()
    
    results = []
    
    for n_points in n_values:
        print(f"Testing N = {n_points}")
        
        for run_idx in range(args.runs):
            print(f"  Run {run_idx + 1}/{args.runs}")
            
            # Generate random data for this run
            run_seed = args.seed + run_idx + n_points if args.seed is not None else None
            X, Y = generate_random_data(n_points, fixed_dim, run_seed)
            
            row = {
                'experiment': 'varying_n',
                'n_points': n_points,
                'dim': fixed_dim,
                'run': run_idx + 1,
                'seed': run_seed,
                'case': 'random'
            }
            
            # Run TSI
            try:
                tsi_score, tsi_time = benchmark_tsi(X, Y)
                row['TSI_score'] = tsi_score
                row['TSI_time'] = tsi_time
                print(f"    TSI: score={tsi_score:.6f}, time={tsi_time:.4f}s")
            except Exception as e:
                print(f"    TSI failed: {e}")
                row['TSI_score'] = np.nan
                row['TSI_time'] = np.nan
            
            # Run baselines
            try:
                baseline_results = benchmark_baselines(X, Y)
                for measure_name, (score, time_taken) in baseline_results.items():
                    # Clean measure name for column naming
                    clean_name = measure_name.replace('/', '_').replace('=', '_').replace('-', '_')
                    row[f'{clean_name}_score'] = score
                    row[f'{clean_name}_time'] = time_taken
                
                print(f"    Baselines completed: {len(baseline_results)} measures")
            except Exception as e:
                print(f"    Baselines failed: {e}")
            
            results.append(row)
        
        print()
    
    return results


def run_d_experiment(args):
    """Run experiment varying D with fixed N=2000."""
    print("=== EXPERIMENT 2: Varying D (fixed N=2000) ===")
    
    d_values = list(range(args.dim_min, args.dim_max + 1, args.dim_step))
    fixed_n = args.fixed_n
    
    print(f"D values: {d_values}")
    print(f"Fixed N: {fixed_n}")
    print(f"Runs per D: {args.runs}")
    print()
    
    results = []
    
    for dim in d_values:
        print(f"Testing D = {dim}")
        
        for run_idx in range(args.runs):
            print(f"  Run {run_idx + 1}/{args.runs}")
            
            # Generate random data for this run
            run_seed = args.seed + run_idx + dim * 10000 if args.seed is not None else None
            X, Y = generate_random_data(fixed_n, dim, run_seed)
            
            row = {
                'experiment': 'varying_d',
                'n_points': fixed_n,
                'dim': dim,
                'run': run_idx + 1,
                'seed': run_seed,
                'case': 'random'
            }
            
            # Run TSI
            try:
                tsi_score, tsi_time = benchmark_tsi(X, Y)
                row['TSI_score'] = tsi_score
                row['TSI_time'] = tsi_time
                print(f"    TSI: score={tsi_score:.6f}, time={tsi_time:.4f}s")
            except Exception as e:
                print(f"    TSI failed: {e}")
                row['TSI_score'] = np.nan
                row['TSI_time'] = np.nan
            
            # Run baselines
            try:
                baseline_results = benchmark_baselines(X, Y)
                for measure_name, (score, time_taken) in baseline_results.items():
                    # Clean measure name for column naming
                    clean_name = measure_name.replace('/', '_').replace('=', '_').replace('-', '_')
                    row[f'{clean_name}_score'] = score
                    row[f'{clean_name}_time'] = time_taken
                
                print(f"    Baselines completed: {len(baseline_results)} measures")
            except Exception as e:
                print(f"    Baselines failed: {e}")
            
            results.append(row)
        
        print()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Random case benchmark: two experiments - varying N (fixed D) and varying D (fixed N)'
    )
    # Dataset parameters for experiment 1 (varying N)
    parser.add_argument('--n-points-min', type=int, default=1000, help='Minimum number of points for varying N experiment')
    parser.add_argument('--n-points-max', type=int, default=2000, help='Maximum number of points for varying N experiment')
    parser.add_argument('--n-points-step', type=int, default=100, help='Step size for number of points in varying N experiment')
    parser.add_argument('--fixed-d', type=int, default=500, help='Fixed dimensionality for varying N experiment')
    
    # Dataset parameters for experiment 2 (varying D)
    parser.add_argument('--dim-min', type=int, default=50, help='Minimum dimensionality for varying D experiment')
    parser.add_argument('--dim-max', type=int, default=500, help='Maximum dimensionality for varying D experiment')
    parser.add_argument('--dim-step', type=int, default=50, help='Step size for dimensionality in varying D experiment')
    parser.add_argument('--fixed-n', type=int, default=2000, help='Fixed number of points for varying D experiment')
    
    # General parameters
    parser.add_argument('--seed', type=int, default=0, help='RNG seed')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs per configuration (for averaging)')
    parser.add_argument('--experiment', choices=['n', 'd', 'both'], default='both', 
                       help='Which experiment to run: n (varying N), d (varying D), or both')

    args = parser.parse_args()

    print(f"Random Case Benchmark:")
    print(f"RNG seed: {args.seed}")
    print(f"Runs per configuration: {args.runs}")
    print(f"Running experiment(s): {args.experiment}")
    print()

    all_results = []

    # Run experiments based on selection
    if args.experiment in ['n', 'both']:
        n_results = run_n_experiment(args)
        all_results.extend(n_results)

    if args.experiment in ['d', 'both']:
        d_results = run_d_experiment(args)
        all_results.extend(d_results)

    # Create DataFrame and save results
    df = pd.DataFrame(all_results)

    # Save CSV
    results_dir = Path(__file__).parent.parent / 'results/benchmark_random'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if args.experiment == 'both':
        filename = f"random_benchmark_both_experiments_runs{args.runs}_seed{args.seed}.csv"
    elif args.experiment == 'n':
        filename = f"random_benchmark_varying_n_n{args.n_points_min}-{args.n_points_max}_d{args.fixed_d}_runs{args.runs}_seed{args.seed}.csv"
    else:  # experiment == 'd'
        filename = f"random_benchmark_varying_d_d{args.dim_min}-{args.dim_max}_n{args.fixed_n}_runs{args.runs}_seed{args.seed}.csv"
    
    filepath = results_dir / filename
    df.to_csv(filepath, index=False)
    print(f"Results saved to: {filepath}")

    # Summary statistics
    print("\n=== SUMMARY ===")
    
    score_columns = [col for col in df.columns if col.endswith('_score')]
    
    if 'varying_n' in df['experiment'].values:
        print(f"\nVARYING N EXPERIMENT (D={args.fixed_d}):")
        n_data = df[df['experiment'] == 'varying_n']
        n_summary = n_data.groupby('n_points')[score_columns].mean().reset_index()
        print("Average scores across runs:")
        print(n_summary.to_string(index=False, float_format='%.6f'))
        
        print("\nTiming summary for varying N:")
        time_columns = [col for col in df.columns if col.endswith('_time')]
        n_time_summary = n_data.groupby('n_points')[time_columns].mean().reset_index()
        print(n_time_summary.to_string(index=False, float_format='%.4f'))
    
    if 'varying_d' in df['experiment'].values:
        print(f"\nVARYING D EXPERIMENT (N={args.fixed_n}):")
        d_data = df[df['experiment'] == 'varying_d']
        d_summary = d_data.groupby('dim')[score_columns].mean().reset_index()
        print("Average scores across runs:")
        print(d_summary.to_string(index=False, float_format='%.6f'))
        
        print("\nTiming summary for varying D:")
        time_columns = [col for col in df.columns if col.endswith('_time')]
        d_time_summary = d_data.groupby('dim')[time_columns].mean().reset_index()
        print(d_time_summary.to_string(index=False, float_format='%.4f'))
    
    print(f"\nTotal measurements: {len(all_results)}")
    if args.experiment == 'both':
        n_measurements = len([r for r in all_results if r['experiment'] == 'varying_n'])
        d_measurements = len([r for r in all_results if r['experiment'] == 'varying_d'])
        print(f"  Varying N experiment: {n_measurements} measurements")
        print(f"  Varying D experiment: {d_measurements} measurements")
    
    baseline_count = len([col for col in score_columns if not col.startswith('TSI')])
    print(f"Measures tested: TSI + {baseline_count} baselines")


if __name__ == "__main__":
    main() 