#!/usr/bin/env python3
"""
Benchmark script for measuring the ability of similarity measures to detect noise.
Tests how noise affects similarity measures by comparing X vs Y(a) where Y(a) = (1-a) * X + a * Noise.
The parameter 'a' controls the amount of noise: a=0 means no noise (Y=X), a=1 means pure noise.
"""

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tsi.baselines import run_baseline_measures
from tsi.tsi import EfficientTSI, RepresentationPair


def generate_data(n_points: int, dim: int = 500, seed: int | None = None) -> np.ndarray:
    """Generate random data for benchmarking."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_points, dim))
    return X


def generate_noise(n_points: int, dim: int, seed: int | None = None) -> np.ndarray:
    """
    Generate noise with the same distribution as X (uniform between 0 and 1).
    
    Args:
        n_points: Number of data points
        dim: Dimensionality
        seed: Random seed for reproducibility
    
    Returns:
        Noise matrix of shape (n_points, dim)
    """
    rng = np.random.default_rng(seed)
    # Generate uniform noise in range [0, 1] to match X distribution
    noise = rng.random((n_points, dim))
    return noise


def create_noisy_data(X: np.ndarray, noise_level: float, seed: int | None = None) -> np.ndarray:
    """
    Create noisy version of X using the formula: Y(a) = (1-a) * X + a * Noise
    
    Args:
        X: Original data matrix
        noise_level: Noise level 'a' (0 = no noise, 1 = pure noise)
        seed: Random seed for reproducibility
    
    Returns:
        Y: Noisy version of X
    """
    if noise_level == 0:
        return X.copy()
    
    n_points, dim = X.shape
    noise = generate_noise(n_points, dim, seed)
    
    # Y(a) = (1-a) * X + a * Noise
    Y = (1 - noise_level) * X + noise_level * noise
    
    return Y


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


def run_noise_experiment(args):
    """Run experiment varying noise level with fixed N."""
    experiment_name = 'varying_noise_uniform'
    
    print(f"=== NOISE IDENTIFICATION EXPERIMENT: Varying Noise Level with Uniform Noise (fixed N={args.fixed_n}) ===")
    
    # Generate noise levels from 0 to 1
    noise_levels = np.linspace(args.noise_min, args.noise_max, args.noise_steps)
    fixed_n = args.fixed_n
    
    print(f"Noise levels (a): {noise_levels}")
    print(f"Fixed N: {fixed_n}")
    print(f"Dimensionality: {args.dim}")
    print(f"Runs per noise level: {args.runs}")
    print(f"Noise type: Uniform [0,1] (same as X)")
    print()
    
    results = []
    
    for noise_level in noise_levels:
        print(f"Testing noise level a = {noise_level:.3f}")
        
        for run_idx in range(args.runs):
            print(f"  Run {run_idx + 1}/{args.runs}")
            
            # Generate data for this run
            run_seed = args.seed + run_idx + int(noise_level * 10000) if args.seed is not None else None
            X = generate_data(fixed_n, args.dim, run_seed)
            
            # Create noisy version Y(a) = (1-a) * X + a * Noise
            noise_seed = run_seed + 50000 if run_seed is not None else None
            Y = create_noisy_data(X, noise_level, noise_seed)
            
            row = {
                'experiment': experiment_name,
                'n_points': fixed_n,
                'dim': args.dim,
                'noise_level': noise_level,
                'noise_type': 'uniform',
                'run': run_idx + 1,
                'seed': run_seed,
                'case': 'noise_uniform'
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
        description='Noise identification benchmark: test how similarity measures detect noise by varying noise level a in Y(a) = (1-a)*X + a*Noise'
    )
    
    # Dataset parameters
    parser.add_argument('--noise-min', type=float, default=0.0, help='Minimum noise level (a) for experiment')
    parser.add_argument('--noise-max', type=float, default=1.0, help='Maximum noise level (a) for experiment')
    parser.add_argument('--noise-steps', type=int, default=21, help='Number of noise level steps (evenly spaced)')
    parser.add_argument('--fixed-n', type=int, default=2000, help='Fixed number of points for experiment')
    parser.add_argument('--dim', type=int, default=500, help='Dimensionality of the data')
    
    # General parameters
    parser.add_argument('--seed', type=int, default=0, help='RNG seed')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs per configuration (for averaging)')

    args = parser.parse_args()

    print(f"Noise Identification Benchmark:")
    print(f"Formula: Y(a) = (1-a) * X + a * Noise")
    print(f"Noise: Uniform [0,1] distribution (same as X)")
    print(f"RNG seed: {args.seed}")
    print(f"Runs per configuration: {args.runs}")
    print()

    all_results = []
    
    # Run the noise experiment
    noise_results = run_noise_experiment(args)
    all_results.extend(noise_results)

    # Create DataFrame and save results
    df = pd.DataFrame(all_results)

    # Save CSV
    results_dir = Path(__file__).parent.parent / 'results/benchmark_noise_identification'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"noise_benchmark_uniform_noise{args.noise_min}-{args.noise_max}_n{args.fixed_n}_steps{args.noise_steps}_runs{args.runs}_seed{args.seed}.csv"
    
    filepath = results_dir / filename
    df.to_csv(filepath, index=False)
    print(f"Results saved to: {filepath}")

    # Summary statistics
    print("\n=== SUMMARY ===")
    
    score_columns = [col for col in df.columns if col.endswith('_score')]
    
    experiment_name = 'varying_noise_uniform'
    if experiment_name in df['experiment'].values:
        print(f"\nUNIFORM NOISE EXPERIMENT (N={args.fixed_n}):")
        noise_data = df[df['experiment'] == experiment_name]
        noise_summary = noise_data.groupby('noise_level')[score_columns].mean().reset_index()
        print("Average scores across runs:")
        print(noise_summary.to_string(index=False, float_format='%.6f'))
        
        print(f"\nTiming summary for uniform noise:")
        time_columns = [col for col in df.columns if col.endswith('_time')]
        noise_time_summary = noise_data.groupby('noise_level')[time_columns].mean().reset_index()
        print(noise_time_summary.to_string(index=False, float_format='%.4f'))
    
    print(f"\nTotal measurements: {len(all_results)}")
    
    baseline_count = len([col for col in score_columns if not col.startswith('TSI')])
    print(f"Measures tested: TSI + {baseline_count} baselines")
    
    print(f"\nNoise level interpretation:")
    print(f"  a = 0.0: No noise (Y = X, perfect similarity expected)")
    print(f"  a = 1.0: Pure noise (Y = Noise, low similarity expected)")
    print(f"  0 < a < 1: Mixed signal (Y = (1-a)*X + a*Noise)")


if __name__ == "__main__":
    main()
