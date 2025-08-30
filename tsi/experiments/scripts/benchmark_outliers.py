#!/usr/bin/env python3
"""
Benchmark script for measuring the impact of outliers on similarity measures.
Tests how outliers affect similarity measures by comparing X vs X' where X' has k randomly selected outliers.
Includes two experiments:
1. Fixed N, varying standard deviation units (0, 1, 2, ..., 10) with PC direction
2. Fixed N, varying standard deviation units (0, 1, 2, ..., 10) with random direction
"""

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tsi.baselines import run_baseline_measures
from tsi.tsi import EfficientTSI, RepresentationPair


def generate_data(n_points: int, dim: int = 500, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate random data for benchmarking."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_points, dim))
    Y = X.copy()  # Start with identical data
    return X, Y


def top_pc_direction(Y: np.ndarray) -> np.ndarray:
    """Compute top principal component direction of Y."""
    Yc = Y - Y.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Yc, full_matrices=False)
    v = Vt[0]
    v /= (np.linalg.norm(v) + 1e-12)
    return v


def random_direction(dim: int, seed: int | None = None) -> np.ndarray:
    """Generate a random unit direction vector."""
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 1, dim)
    v /= (np.linalg.norm(v) + 1e-12)
    return v


def translate_subset(Y: np.ndarray, indices: np.ndarray, direction: np.ndarray, alpha: float) -> np.ndarray:
    """Return a copy of Y where rows[indices] are moved by alpha * direction."""
    Yp = Y.copy()
    Yp[indices] = Yp[indices] + alpha * direction
    return Yp


def create_outliers(X: np.ndarray, Y: np.ndarray, k_outliers: int, sigma: float, seed: int | None = None) -> np.ndarray:
    """
    Create outliers by translating k randomly selected points in Y by sigma standard deviations
    along the top principal component direction.
    
    Args:
        X: Original data
        Y: Data to modify (should be copy of X)
        k_outliers: Number of outliers to create
        sigma: Number of standard deviations to translate points
        seed: Random seed for reproducibility
    
    Returns:
        Y_prime: Modified Y with outliers
    """
    if k_outliers == 0:
        return Y.copy()
    
    rng = np.random.default_rng(seed)
    
    # Get top principal component direction
    direction = top_pc_direction(Y)
    
    # Randomly select indices for outliers
    n_points = len(Y)
    outlier_indices = rng.choice(n_points, size=min(k_outliers, n_points), replace=False)
    
    # Translate outliers by sigma standard deviations
    Y_prime = translate_subset(Y, outlier_indices, direction, sigma)
    
    return Y_prime


def create_outliers_random_direction(X: np.ndarray, Y: np.ndarray, k_outliers: int, sigma: float, seed: int | None = None) -> np.ndarray:
    """
    Create outliers by translating k randomly selected points in Y by sigma standard deviations
    along a random direction.
    
    Args:
        X: Original data
        Y: Data to modify (should be copy of X)
        k_outliers: Number of outliers to create
        sigma: Number of standard deviations to translate points
        seed: Random seed for reproducibility
    
    Returns:
        Y_prime: Modified Y with outliers
    """
    if k_outliers == 0:
        return Y.copy()
    
    rng = np.random.default_rng(seed)
    
    # Get random direction
    direction = random_direction(Y.shape[1], seed)
    
    # Randomly select indices for outliers
    n_points = len(Y)
    outlier_indices = rng.choice(n_points, size=min(k_outliers, n_points), replace=False)
    
    # Translate outliers by sigma standard deviations
    Y_prime = translate_subset(Y, outlier_indices, direction, sigma)
    
    return Y_prime


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


def run_sigma_experiment(args, direction_type='pc'):
    """Run experiment varying sigma (standard deviations) with fixed N using specified direction type."""
    direction_name = "PC Direction" if direction_type == 'pc' else "Random Direction"
    experiment_num = "1" if direction_type == 'pc' else "2"
    
    print(f"=== EXPERIMENT {experiment_num}: Varying Sigma with {direction_name} (fixed N={args.fixed_n}, k={args.k_outliers}) ===")
    
    sigma_values = list(range(args.sigma_min, args.sigma_max + 1, args.sigma_step))
    fixed_n = args.fixed_n
    
    print(f"Sigma values: {sigma_values}")
    print(f"Fixed N: {fixed_n}")
    print(f"Number of outliers (k): {args.k_outliers}")
    print(f"Dimensionality: {args.dim}")
    print(f"Runs per sigma: {args.runs}")
    print(f"Outlier direction: {direction_name}")
    print()
    
    results = []
    
    for sigma in sigma_values:
        print(f"Testing sigma = {sigma}")
        
        for run_idx in range(args.runs):
            print(f"  Run {run_idx + 1}/{args.runs}")
            
            # Generate data for this run
            run_seed = args.seed + run_idx + sigma * 1000 if args.seed is not None else None
            X, Y = generate_data(fixed_n, args.dim, run_seed)
            
            # Create outliers based on direction type
            outlier_seed = run_seed + 10000 if run_seed is not None else None
            if direction_type == 'pc':
                Y_prime = create_outliers(X, Y, args.k_outliers, sigma, outlier_seed)
                experiment_name = 'varying_sigma_pc'
                case_name = 'outliers_pc_direction'
            else:  # direction_type == 'random'
                Y_prime = create_outliers_random_direction(X, Y, args.k_outliers, sigma, outlier_seed)
                experiment_name = 'varying_sigma_random'
                case_name = 'outliers_random_direction'
            
            row = {
                'experiment': experiment_name,
                'n_points': fixed_n,
                'dim': args.dim,
                'k_outliers': args.k_outliers,
                'sigma': sigma,
                'run': run_idx + 1,
                'seed': run_seed,
                'case': case_name
            }
            
            # Run TSI
            try:
                tsi_score, tsi_time = benchmark_tsi(X, Y_prime)
                row['TSI_score'] = tsi_score
                row['TSI_time'] = tsi_time
                print(f"    TSI: score={tsi_score:.6f}, time={tsi_time:.4f}s")
            except Exception as e:
                print(f"    TSI failed: {e}")
                row['TSI_score'] = np.nan
                row['TSI_time'] = np.nan
            
            # Run baselines
            try:
                baseline_results = benchmark_baselines(X, Y_prime)
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
        description='Outlier impact benchmark: two experiments - varying sigma with PC direction and varying sigma with random direction'
    )
    
    # Dataset parameters for both experiments (both vary sigma with fixed N)
    parser.add_argument('--sigma-min', type=int, default=0, help='Minimum sigma (std deviations) for varying sigma experiments')
    parser.add_argument('--sigma-max', type=int, default=100, help='Maximum sigma (std deviations) for varying sigma experiments')
    parser.add_argument('--sigma-step', type=int, default=5, help='Step size for sigma in varying sigma experiments')
    parser.add_argument('--fixed-n', type=int, default=2000, help='Fixed number of points for both experiments')
    
    # Outlier parameters
    parser.add_argument('--k-outliers', type=int, default=20, help='Number of outliers to create')
    parser.add_argument('--dim', type=int, default=500, help='Dimensionality of the data')
    
    # General parameters
    parser.add_argument('--seed', type=int, default=0, help='RNG seed')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs per configuration (for averaging)')
    parser.add_argument('--experiment', choices=['pc', 'random', 'both'], default='both', 
                       help='Which experiment to run: pc (PC direction), random (random direction), or both')

    args = parser.parse_args()

    print(f"Outlier Impact Benchmark:")
    print(f"RNG seed: {args.seed}")
    print(f"Runs per configuration: {args.runs}")
    print(f"Running experiment(s): {args.experiment}")
    print()

    all_results = []

    # Run experiments based on selection
    if args.experiment in ['pc', 'both']:
        pc_results = run_sigma_experiment(args, direction_type='pc')
        all_results.extend(pc_results)

    if args.experiment in ['random', 'both']:
        random_results = run_sigma_experiment(args, direction_type='random')
        all_results.extend(random_results)

    # Create DataFrame and save results
    df = pd.DataFrame(all_results)

    # Save CSV
    results_dir = Path(__file__).parent.parent / 'results/benchmark_outliers'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if args.experiment == 'both':
        filename = f"outliers_benchmark_both_directions_k{args.k_outliers}_runs{args.runs}_seed{args.seed}.csv"
    elif args.experiment == 'pc':
        filename = f"outliers_benchmark_pc_direction_sigma{args.sigma_min}-{args.sigma_max}_n{args.fixed_n}_k{args.k_outliers}_runs{args.runs}_seed{args.seed}.csv"
    else:  # experiment == 'random'
        filename = f"outliers_benchmark_random_direction_sigma{args.sigma_min}-{args.sigma_max}_n{args.fixed_n}_k{args.k_outliers}_runs{args.runs}_seed{args.seed}.csv"
    
    filepath = results_dir / filename
    df.to_csv(filepath, index=False)
    print(f"Results saved to: {filepath}")

    # Summary statistics
    print("\n=== SUMMARY ===")
    
    score_columns = [col for col in df.columns if col.endswith('_score')]
    
    if 'varying_sigma_pc' in df['experiment'].values:
        print(f"\nPC DIRECTION EXPERIMENT (N={args.fixed_n}, k={args.k_outliers}):")
        pc_data = df[df['experiment'] == 'varying_sigma_pc']
        pc_summary = pc_data.groupby('sigma')[score_columns].mean().reset_index()
        print("Average scores across runs:")
        print(pc_summary.to_string(index=False, float_format='%.6f'))
        
        print("\nTiming summary for PC direction:")
        time_columns = [col for col in df.columns if col.endswith('_time')]
        pc_time_summary = pc_data.groupby('sigma')[time_columns].mean().reset_index()
        print(pc_time_summary.to_string(index=False, float_format='%.4f'))
    
    if 'varying_sigma_random' in df['experiment'].values:
        print(f"\nRANDOM DIRECTION EXPERIMENT (N={args.fixed_n}, k={args.k_outliers}):")
        random_data = df[df['experiment'] == 'varying_sigma_random']
        random_summary = random_data.groupby('sigma')[score_columns].mean().reset_index()
        print("Average scores across runs:")
        print(random_summary.to_string(index=False, float_format='%.6f'))
        
        print("\nTiming summary for random direction:")
        time_columns = [col for col in df.columns if col.endswith('_time')]
        random_time_summary = random_data.groupby('sigma')[time_columns].mean().reset_index()
        print(random_time_summary.to_string(index=False, float_format='%.4f'))
    
    print(f"\nTotal measurements: {len(all_results)}")
    if args.experiment == 'both':
        pc_measurements = len([r for r in all_results if r['experiment'] == 'varying_sigma_pc'])
        random_measurements = len([r for r in all_results if r['experiment'] == 'varying_sigma_random'])
        print(f"  PC direction experiment: {pc_measurements} measurements")
        print(f"  Random direction experiment: {random_measurements} measurements")
    
    baseline_count = len([col for col in score_columns if not col.startswith('TSI')])
    print(f"Measures tested: TSI + {baseline_count} baselines")


if __name__ == "__main__":
    main()