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
from src.baselines import run_baseline_measures
from src.tsi import EfficientTSI
from src.qsi import EfficientQSI
from src.data import RepresentationPair


def generate_data(n_points: int, dim: int = 500, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate random data for benchmarking."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_points, dim))
    Y = X.copy()  # Start with identical data
    return X, Y


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


def create_outliers_per_outlier_random_direction(X: np.ndarray, Y: np.ndarray, k_outliers: int, sigma: float, seed: int | None = None) -> np.ndarray:
    """
    Create outliers by translating k randomly selected points in Y, where each selected point
    is moved along its own independently sampled random direction (k random directions total).
    """
    if k_outliers == 0:
        return Y.copy()

    rng = np.random.default_rng(seed)

    n_points, dim = Y.shape
    outlier_indices = rng.choice(n_points, size=min(k_outliers, n_points), replace=False)

    Y_prime = Y.copy()
    for idx in outlier_indices:
        v = rng.normal(0, 1, dim)
        v /= (np.linalg.norm(v) + 1e-12)
        Y_prime[idx] = Y_prime[idx] + sigma * v

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


def benchmark_qsi(X: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
    """Benchmark QSI computation time."""
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    efficient_qsi = EfficientQSI(euclidean=True)
    start_time = time.time()
    score = efficient_qsi(representations)
    end_time = time.time()
    return score, end_time - start_time


def benchmark_baselines(X: np.ndarray, Y: np.ndarray) -> dict:
    """Benchmark baseline measures computation time."""
    return run_baseline_measures(X, Y, time_monitor=True)


def run_sigma_experiment(args):
    """Run experiment varying sigma (standard deviations) with fixed N using random direction."""
    print(f"=== EXPERIMENT: Varying Sigma with Random Direction (fixed N={args.fixed_n}, k={args.k_outliers}) ===")
    
    # Build multiplicative sigma schedule: 0, 1, factor, factor^2, ... <= sigma_max
    sigma_values = [0]
    current_sigma = 1
    while current_sigma <= args.sigma_max:
        sigma_values.append(int(current_sigma))
        next_sigma = int(current_sigma * args.sigma_step)
        if next_sigma <= current_sigma:
            break
        current_sigma = next_sigma
    fixed_n = args.fixed_n
    
    print(f"Sigma values: {sigma_values}")
    print(f"Fixed N: {fixed_n}")
    print(f"Number of outliers (k): {args.k_outliers}")
    print(f"Dimensionality: {args.dim}")
    print(f"Runs per sigma: {args.runs}")
    print(f"Outlier direction: Random Direction")
    print()
    
    results = []
    
    for sigma in sigma_values:
        print(f"Testing sigma = {sigma}")
        
        for run_idx in range(args.runs):
            print(f"  Run {run_idx + 1}/{args.runs}")
            
            # Generate data for this run
            run_seed = args.seed + run_idx + sigma * 1000 if args.seed is not None else None
            X, Y = generate_data(fixed_n, args.dim, run_seed)
            
            # Create outliers in random direction according to selected mode
            outlier_seed = run_seed + 10000 if run_seed is not None else None
            if args.outlier_direction_mode == 'single':
                Y_prime = create_outliers_random_direction(X, Y, args.k_outliers, sigma, outlier_seed)
                mode_suffix = 'single'
            else:  # 'per_outlier'
                Y_prime = create_outliers_per_outlier_random_direction(X, Y, args.k_outliers, sigma, outlier_seed)
                mode_suffix = 'per_outlier'
            experiment_name = 'varying_sigma_random'
            case_name = f'outliers_random_direction_{mode_suffix}'
            
            row = {
                'experiment': experiment_name,
                'n_points': fixed_n,
                'dim': args.dim,
                'k_outliers': args.k_outliers,
                'sigma': sigma,
                'run': run_idx + 1,
                'seed': run_seed,
                'case': case_name,
                'outlier_direction_mode': args.outlier_direction_mode
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
            
            # Run QSI
            try:
                qsi_score, qsi_time = benchmark_qsi(X, Y_prime)
                row['QSI_score'] = qsi_score
                row['QSI_time'] = qsi_time
                print(f"    QSI: score={qsi_score:.6f}, time={qsi_time:.4f}s")
            except Exception as e:
                print(f"    QSI failed: {e}")
                row['QSI_score'] = np.nan
                row['QSI_time'] = np.nan

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
        description='Outlier impact benchmark: varying sigma with random direction only'
    )
    
    # Dataset parameters (vary sigma with fixed N)
    parser.add_argument('--sigma-min', type=int, default=0, help='Minimum sigma (std deviations) for varying sigma experiments')
    parser.add_argument('--sigma-max', type=int, default=128, help='Maximum sigma (std deviations) for varying sigma experiments')
    parser.add_argument('--sigma-step', type=int, default=2, help='Multiplicative factor for sigma after initial 0 and 1 (e.g., 2 gives 0,1,2,4,8,...)')
    parser.add_argument('--fixed-n', type=int, default=1000, help='Fixed number of points for both experiments')
    
    # Outlier parameters
    parser.add_argument('--k-outliers', type=int, default=20, help='Number of outliers to create')
    parser.add_argument('--dim', type=int, default=50, help='Dimensionality of the data')
    parser.add_argument('--outlier-direction-mode', choices=['single', 'per_outlier'], default='per_outlier',
                        help='Outlier translation mode: single random direction for all outliers, or per_outlier random directions')
    
    # General parameters
    parser.add_argument('--seed', type=int, default=0, help='RNG seed')
    parser.add_argument('--runs', type=int, default=20, help='Number of runs per configuration (for averaging)')

    args = parser.parse_args()

    print(f"Outlier Impact Benchmark:")
    print(f"RNG seed: {args.seed}")
    print(f"Runs per configuration: {args.runs}")
    print(f"Running experiment: random direction only")
    print(f"Outlier direction mode: {args.outlier_direction_mode}")
    print()

    all_results = run_sigma_experiment(args)

    # Create DataFrame and save results
    df = pd.DataFrame(all_results)

    # Save CSV
    results_dir = Path(__file__).parent.parent / 'results/benchmark_outliers'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"outliers_benchmark_random_direction_{args.outlier_direction_mode}_sigma{args.sigma_min}-{args.sigma_max}_n{args.fixed_n}_k{args.k_outliers}_runs{args.runs}_seed{args.seed}.csv"
    
    filepath = results_dir / filename
    df.to_csv(filepath, index=False)
    print(f"Results saved to: {filepath}")

    # Summary statistics
    print("\n=== SUMMARY ===")
    
    score_columns = [col for col in df.columns if col.endswith('_score')]
    
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
    
    baseline_count = len([col for col in score_columns if not col.startswith('TSI') and not col.startswith('QSI')])
    print(f"Measures tested: TSI + QSI + {baseline_count} baselines")


if __name__ == "__main__":
    main()