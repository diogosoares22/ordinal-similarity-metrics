#!/usr/bin/env python3
"""
Benchmark script for measuring the impact of outliers on similarity measures.
Tests how outliers affect similarity measures by comparing X vs X' where X' has k randomly selected outliers.
Supports two data sources:
1. Synthetic: Random uniform data
2. CIFAR-10: Real neural network representations from trained model

Experiments:
- Fixed N, varying standard deviation units (0, 1, 2, ..., max_sigma) with random direction
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


# Path to CIFAR-10 representations
CIFAR10_REPRESENTATIONS_PATH =  'data/cifar-10-final-epoch-val-representations.npy'


def load_cifar10_representations() -> np.ndarray:
    """Load CIFAR-10 final epoch validation representations."""
    return np.load(CIFAR10_REPRESENTATIONS_PATH)


def generate_synthetic_data(n_points: int, dim: int = 500, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate random synthetic data for benchmarking."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_points, dim))
    Y = X.copy()  # Start with identical data
    return X, Y


def generate_cifar10_data(n_points: int | None = None, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Load CIFAR-10 representations and prepare X, Y pair.
    
    Args:
        n_points: Number of points to use. If None, uses all available.
        seed: Random seed for shuffling/subsampling.
    
    Returns:
        X, Y tuple where Y is a copy of X (identical data)
    """
    X_full = load_cifar10_representations()
    rng = np.random.default_rng(seed)
    
    if n_points is not None and n_points < len(X_full):
        # Randomly subsample
        indices = rng.choice(len(X_full), size=n_points, replace=False)
        X = X_full[indices]
    else:
        X = X_full.copy()
    
    Y = X.copy()  # Start with identical data
    return X, Y


def compute_avg_pairwise_distance(X: np.ndarray) -> float:
    """
    Compute the average pairwise L2 distance between all points in X.
    
    Args:
        X: Data matrix of shape (n_points, dim)
    
    Returns:
        Average pairwise L2 distance
    """
    from scipy.spatial.distance import pdist
    distances = pdist(X, metric='euclidean')
    return float(np.mean(distances))


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


def create_outliers_random_direction(X: np.ndarray, Y: np.ndarray, k_outliers: int, sigma: float, 
                                      avg_distance: float, seed: int | None = None) -> np.ndarray:
    """
    Create outliers by translating k randomly selected points in Y by sigma * avg_distance
    along a random direction.
    
    Args:
        X: Original data
        Y: Data to modify (should be copy of X)
        k_outliers: Number of outliers to create
        sigma: Multiplier for average distance (outlier strength)
        avg_distance: Average pairwise L2 distance in the dataset (scaling factor)
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
    
    # Translate outliers by sigma * avg_distance
    translation_magnitude = sigma * avg_distance
    Y_prime = translate_subset(Y, outlier_indices, direction, translation_magnitude)
    
    return Y_prime


def create_outliers_per_outlier_random_direction(X: np.ndarray, Y: np.ndarray, k_outliers: int, sigma: float,
                                                   avg_distance: float, seed: int | None = None) -> np.ndarray:
    """
    Create outliers by translating k randomly selected points in Y, where each selected point
    is moved along its own independently sampled random direction (k random directions total).
    Translation magnitude is sigma * avg_distance.
    
    Args:
        X: Original data
        Y: Data to modify (should be copy of X)
        k_outliers: Number of outliers to create
        sigma: Multiplier for average distance (outlier strength)
        avg_distance: Average pairwise L2 distance in the dataset (scaling factor)
        seed: Random seed for reproducibility
    
    Returns:
        Y_prime: Modified Y with outliers
    """
    if k_outliers == 0:
        return Y.copy()

    rng = np.random.default_rng(seed)

    n_points, dim = Y.shape
    outlier_indices = rng.choice(n_points, size=min(k_outliers, n_points), replace=False)

    # Compute translation magnitude
    translation_magnitude = sigma * avg_distance

    Y_prime = Y.copy()
    for idx in outlier_indices:
        v = rng.normal(0, 1, dim)
        v /= (np.linalg.norm(v) + 1e-12)
        Y_prime[idx] = Y_prime[idx] + translation_magnitude * v

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


def run_sigma_experiment(args, data_source: str):
    """Run experiment varying sigma (standard deviations) with fixed N using random direction.
    
    Args:
        args: Command line arguments
        data_source: Either 'synthetic' or 'cifar10'
    """
    data_source_display = 'CIFAR-10' if data_source == 'cifar10' else 'Synthetic'
    
    # Determine N and dim based on data source
    if data_source == 'cifar10':
        # For CIFAR-10, we use all 10k samples with dim=512
        fixed_n = args.cifar10_n if args.cifar10_n else 10000  # Default to all samples
        dim = 512  # Fixed dimension for CIFAR-10 representations
    else:
        fixed_n = args.fixed_n
        dim = args.dim
    
    # Calculate number of outliers from percentage
    k_outliers = max(1, int(fixed_n * args.outlier_pct / 100.0))
    
    print(f"=== EXPERIMENT: Varying Sigma with Random Direction ({data_source_display}) ===")
    print(f"  N={fixed_n}, k={k_outliers} ({args.outlier_pct}%), dim={dim}")
    
    # Build multiplicative sigma schedule: 0, 1, factor, factor^2, ... <= sigma_max
    sigma_values = [0]
    current_sigma = 1
    while current_sigma <= args.sigma_max:
        sigma_values.append(int(current_sigma))
        next_sigma = int(current_sigma * args.sigma_step)
        if next_sigma <= current_sigma:
            break
        current_sigma = next_sigma
    
    print(f"Sigma values: {sigma_values}")
    print(f"Number of outliers (k): {k_outliers} ({args.outlier_pct}% of N)")
    print(f"Runs per sigma: {args.runs}")
    print(f"Outlier direction: Random Direction")
    print()
    
    # Compute average pairwise distance once for scaling outlier translations
    # Generate a reference dataset to compute the average distance
    if data_source == 'cifar10':
        X_ref, _ = generate_cifar10_data(n_points=fixed_n, seed=args.seed)
    else:
        X_ref, _ = generate_synthetic_data(fixed_n, dim, args.seed)
    
    avg_distance = compute_avg_pairwise_distance(X_ref)
    print(f"Average pairwise L2 distance: {avg_distance:.4f}")
    print()
    
    results = []
    
    for sigma in sigma_values:
        print(f"Testing sigma = {sigma}")
        
        for run_idx in range(args.runs):
            print(f"  Run {run_idx + 1}/{args.runs}")
            
            # Generate data for this run
            run_seed = args.seed + run_idx + sigma * 1000 if args.seed is not None else None
            
            if data_source == 'cifar10':
                X, Y = generate_cifar10_data(n_points=fixed_n, seed=run_seed)
            else:
                X, Y = generate_synthetic_data(fixed_n, dim, run_seed)
            
            # Create outliers in random direction according to selected mode
            outlier_seed = run_seed + 10000 if run_seed is not None else None
            if args.outlier_direction_mode == 'single':
                Y_prime = create_outliers_random_direction(X, Y, k_outliers, sigma, avg_distance, outlier_seed)
                mode_suffix = 'single'
            else:  # 'per_outlier'
                Y_prime = create_outliers_per_outlier_random_direction(X, Y, k_outliers, sigma, avg_distance, outlier_seed)
                mode_suffix = 'per_outlier'
            
            experiment_name = f'varying_sigma_random_{data_source}'
            case_name = f'outliers_random_direction_{mode_suffix}'
            
            row = {
                'experiment': experiment_name,
                'data_source': data_source,
                'n_points': len(X),
                'dim': X.shape[1],
                'outlier_pct': args.outlier_pct,
                'k_outliers': k_outliers,
                'sigma': sigma,
                'avg_pairwise_distance': avg_distance,
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
        description='Outlier impact benchmark: varying sigma with random direction for synthetic and CIFAR-10 data'
    )
    
    # Data source selection
    parser.add_argument('--data-sources', nargs='+', choices=['synthetic', 'cifar10', 'both'], default=['both'],
                        help='Data sources to benchmark: synthetic, cifar10, or both')
    
    # Sigma parameters (for both data sources)
    parser.add_argument('--sigma-min', type=int, default=0, help='Minimum sigma (std deviations) for varying sigma experiments')
    parser.add_argument('--sigma-max', type=int, default=128, help='Maximum sigma (std deviations) for varying sigma experiments')
    parser.add_argument('--sigma-step', type=int, default=2, help='Multiplicative factor for sigma after initial 0 and 1 (e.g., 2 gives 0,1,2,4,8,...)')
    
    # Synthetic data parameters
    parser.add_argument('--fixed-n', type=int, default=1000, help='Fixed number of points for synthetic data')
    parser.add_argument('--dim', type=int, default=50, help='Dimensionality of synthetic data')
    
    # CIFAR-10 data parameters
    parser.add_argument('--cifar10-n', type=int, default=None, help='Number of CIFAR-10 samples to use (default: all 10000)')
    
    # Outlier parameters (shared)
    parser.add_argument('--outlier-pct', type=float, default=2.0, help='Percentage of data points to make outliers (default: 2%%)')
    parser.add_argument('--outlier-direction-mode', choices=['single', 'per_outlier'], default='per_outlier',
                        help='Outlier translation mode: single random direction for all outliers, or per_outlier random directions')
    
    # General parameters
    parser.add_argument('--seed', type=int, default=0, help='RNG seed')
    parser.add_argument('--runs', type=int, default=20, help='Number of runs per configuration (for averaging)')

    args = parser.parse_args()
    
    # Normalize data sources
    if 'both' in args.data_sources:
        data_sources = ['synthetic', 'cifar10']
    else:
        data_sources = args.data_sources

    print(f"Outlier Impact Benchmark:")
    print(f"Data sources: {data_sources}")
    print(f"Outlier percentage: {args.outlier_pct}%")
    print(f"RNG seed: {args.seed}")
    print(f"Runs per configuration: {args.runs}")
    print(f"Outlier direction mode: {args.outlier_direction_mode}")
    print()

    all_results = []
    
    for data_source in data_sources:
        print(f"\n{'='*60}")
        print(f"Running experiments with {data_source.upper()} data")
        print(f"{'='*60}\n")
        
        results = run_sigma_experiment(args, data_source)
        all_results.extend(results)

    # Create DataFrame and save results
    df = pd.DataFrame(all_results)

    # Save CSV
    results_dir = Path(__file__).parent.parent / 'results/benchmark_outliers'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename based on data sources used
    data_source_str = '_'.join(sorted(data_sources))
    filename = f"outliers_benchmark_{data_source_str}_{args.outlier_direction_mode}_sigma{args.sigma_min}-{args.sigma_max}_pct{args.outlier_pct}_runs{args.runs}_seed{args.seed}.csv"
    
    filepath = results_dir / filename
    df.to_csv(filepath, index=False)
    print(f"Results saved to: {filepath}")

    # Summary statistics
    print("\n=== SUMMARY ===")
    
    score_columns = [col for col in df.columns if col.endswith('_score')]
    
    for data_source in data_sources:
        source_data = df[df['data_source'] == data_source]
        if len(source_data) == 0:
            continue
            
        source_display = 'CIFAR-10' if data_source == 'cifar10' else 'Synthetic'
        n_points = source_data['n_points'].iloc[0]
        dim = source_data['dim'].iloc[0]
        k_outliers = source_data['k_outliers'].iloc[0]
        avg_dist = source_data['avg_pairwise_distance'].mean()
        
        print(f"\n{source_display.upper()} DATA (N={n_points}, dim={dim}, k={k_outliers} [{args.outlier_pct}%], avg_dist={avg_dist:.4f}):")
        summary = source_data.groupby('sigma')[score_columns].mean().reset_index()
        print("Average scores across runs:")
        print(summary.to_string(index=False, float_format='%.6f'))
        
        print(f"\nTiming summary for {source_display}:")
        time_columns = [col for col in df.columns if col.endswith('_time')]
        time_summary = source_data.groupby('sigma')[time_columns].mean().reset_index()
        print(time_summary.to_string(index=False, float_format='%.4f'))
    
    print(f"\nTotal measurements: {len(all_results)}")
    
    baseline_count = len([col for col in score_columns if not col.startswith('TSI') and not col.startswith('QSI')])
    print(f"Measures tested: TSI + QSI + {baseline_count} baselines")


if __name__ == "__main__":
    main()
