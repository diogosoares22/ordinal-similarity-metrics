#!/usr/bin/env python3
"""
Benchmark script for outlier detection of baseline measures vs TSI.
"""

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tsi.baselines import run_baseline_measures
from tsi.tsi import EfficientTSI, RepresentationPair

# non-interactive plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_data(n_points: int, dim: int, data_init: str, rho: float, seed: int | None):
    """Generate X and Y according to initialization mode."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_points, dim))
    if data_init == "equal":
        Y = X.copy()
    elif data_init == "correlated":
        Xz = X - X.mean(axis=0, keepdims=True)
        Z = rng.standard_normal((n_points, dim))
        Yz = rho * Xz + np.sqrt(max(0.0, 1.0 - rho**2)) * Z
        Y = (Yz - Yz.min()) / (Yz.max() - Yz.min() + 1e-12)
    else:  # independent
        Y = rng.random((n_points, dim))
    return X, Y


def top_pc_direction(Y: np.ndarray) -> np.ndarray:
    """Compute top principal component direction of Y."""
    Yc = Y - Y.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Yc, full_matrices=False)
    v = Vt[0]
    v /= (np.linalg.norm(v) + 1e-12)
    return v


def translate_subset(Y: np.ndarray, indices: np.ndarray, direction: np.ndarray, alpha: float) -> np.ndarray:
    """Return a copy of Y where rows[indices] are moved by alpha * direction."""
    Yp = Y.copy()
    Yp[indices] = Yp[indices] + alpha * direction
    return Yp


def benchmark_tsi(X: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
    """Benchmark TSI computation time (uses EfficientTSI from your package)."""
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    efficient_tsi = EfficientTSI(euclidean=True, memory_efficient=True)
    start_time = time.time()
    score = efficient_tsi(representations)
    end_time = time.time()
    return score, end_time - start_time


def benchmark_baselines(X: np.ndarray, Y: np.ndarray) -> dict:
    """Benchmark baseline measures computation time (delegated to your package)."""
    return run_baseline_measures(X, Y, time_monitor=True)


def main():
    parser = argparse.ArgumentParser(
        description='Outlier (subset-translation) benchmark: baseline measures vs TSI'
    )
    # Sweep over alpha
    parser.add_argument('--xticks', type=int, default=30, help='Number of alpha points to test')
    parser.add_argument('--factor', type=float, default=1.0, help='Additive step for alpha')
    parser.add_argument('--initial', type=float, default=0.0, help='Initial alpha value')
    # Dataset & outliers
    parser.add_argument('--n-points', type=int, default=1000, help='Number of points (fixed for the run)')
    parser.add_argument('--dim', type=int, default=500, help='Dimensionality')
    parser.add_argument('--m-outliers', type=int, default=10, help='Number of points to translate in Y')
    parser.add_argument('--with-tsi', action='store_true', default=False, help='Include TSI')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed')
    # Data init mode
    parser.add_argument('--data-init', choices=['independent', 'equal', 'correlated'],
                        default='independent', help='How to initialize Y relative to X')
    parser.add_argument('--rho', type=float, default=0.95, help='Correlation used when --data-init correlated')
    # NEW: normalize plotting so each curve starts at 1
    parser.add_argument('--normalize-alpha0', action='store_true', default=False,
                        help='Normalize each metric curve so its value at alpha=0 is 1 (plotting only)')

    args = parser.parse_args()

    # Build alpha sweep
    alphas = [args.initial + (args.factor * i) for i in range(args.xticks)]
    print(f"Benchmarking with alpha values: {alphas}")
    print(f"Data init: {args.data_init}{f' (rho={args.rho})' if args.data_init=='correlated' else ''}, seed={args.seed}")

    # Generate base data once (fixed size), choose subset once, direction once
    X, Y = generate_data(args.n_points, args.dim, args.data_init, args.rho, args.seed)
    if args.m_outliers < 1 or args.m_outliers > args.n_points:
        raise ValueError(f"--m-outliers must be in [1, {args.n_points}]")
    rng = np.random.default_rng(args.seed)
    subset_indices = rng.choice(args.n_points, size=args.m_outliers, replace=False)
    direction = top_pc_direction(Y)

    results = []

    for alpha in alphas:
        print(f"Testing alpha={alpha} (m_outliers={args.m_outliers}, n_points={args.n_points})...")

        # Apply subset translation to Y
        Y_perturbed = translate_subset(Y, subset_indices, direction, alpha)

        row = {
            'alpha': alpha,
            'm_outliers': args.m_outliers,
            'n_points': args.n_points,
        }

        # Optional TSI
        if args.with_tsi:
            tsi_score, tsi_time = benchmark_tsi(X, Y_perturbed)
            row['tsi_score'] = tsi_score
            row['tsi_time'] = tsi_time

        # Baselines
        baseline_results = benchmark_baselines(X, Y_perturbed)
        for measure_name, (score, time_taken) in baseline_results.items():
            row[f'{measure_name}_score'] = score
            row[f'{measure_name}_time'] = time_taken

        results.append(row)

        if args.with_tsi:
            print(f"  TSI time: {tsi_time:.4f}s")
        print(f"  Baseline times: {[f'{t:.4f}s' for (_, t) in baseline_results.values()]}")

    df = pd.DataFrame(results)

    # Save CSV
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"outlier_benchmark_xticks{args.xticks}_factor{args.factor}_initial{args.initial}"
        f"_n{args.n_points}_d{args.dim}_m{args.m_outliers}_{args.data_init}.csv"
    )
    filepath = results_dir / filename
    df.to_csv(filepath, index=False)
    print(f"\nResults saved to: {filepath}")

    # -------- Plot: alpha vs selected metrics --------
    # Pick columns heuristically
    cka_col = next((c for c in df.columns if c.endswith('_score') and 'cka' in c and 'rbf' not in c and 'unbiased' not in c), None)
    svcca_col = next((c for c in df.columns if c.endswith('_score') and 'svcca' in c), None)
    mu_col = next((c for c in df.columns if c.endswith('_score') and 'mutual_knn_topk' in c), None)  # FIXED endswith
    lcs_col = next((c for c in df.columns if c.endswith('_score') and 'lcs_knn_topk' in c), None)
    has_tsi = 'tsi_score' in df.columns

    # For normalization: cache alpha=0 row
    alpha0 = df.iloc[(df['alpha'] - alphas[0]).abs().values.argmin()]
    def norm_curve(name, values):
        if not args.normalize_alpha0:
            return values
        v0 = float(alpha0[name])
        if np.isclose(v0, 0.0):
            return values  # avoid divide-by-zero; leave as-is
        # heuristic: if it's a distance-like metric, invert ratio so larger distance -> smaller normalized similarity
        is_distance = ('distance' in name) or ('edit_' in name and 'distance' in name)
        if is_distance:
            return v0 / values
        else:
            return values / v0

    if any([has_tsi, cka_col, svcca_col, mu_col, lcs_col]):
        plt.figure(figsize=(9, 5))
        x = df['alpha'].values
        if has_tsi:
            plt.plot(x, norm_curve('tsi_score', df['tsi_score'].values), marker='o', label='TSI')
        if cka_col:
            plt.plot(x, norm_curve(cka_col, df[cka_col].values), marker='s', label=f'CKA ({cka_col})')
        if svcca_col:
            plt.plot(x, norm_curve(svcca_col, df[svcca_col].values), marker='^', label=f'SVCCA ({svcca_col})')
        if mu_col:
            plt.plot(x, norm_curve(mu_col, df[mu_col].values), marker='>', label=f'MI_KNN ({mu_col})')
        if lcs_col:
            plt.plot(x, norm_curve(lcs_col, df[lcs_col].values), marker='<', label=f'LCS_KNN ({lcs_col})')
        plt.xlabel('Deviation (alpha)')
        plt.ylabel('Similarity score' + (' (normalized to α=0)' if args.normalize_alpha0 else ''))
        title_hint = f"init={args.data_init}" + (f", rho={args.rho}" if args.data_init == 'correlated' else "")
        plt.title(f'Outlier sensitivity: TSI vs CKA vs SVCCA ({title_hint})')
        plt.legend()
        plt.tight_layout()
        plotpath = results_dir / filename.replace('.csv', '_normalized.png' if args.normalize_alpha0 else '.png')
        plt.savefig(plotpath, dpi=150)
        print(f"Plot saved to: {plotpath}")
    else:
        print("No TSI/CKA/SVCCA columns found to plot; skipping figure.")

    # Summary
    print("\nBenchmark Summary:")
    print(f"Alpha values tested: {alphas}")
    print(f"Total measurements: {len(results)}")
    print(f"Subset size (m_outliers): {args.m_outliers}")
    print(f"Dataset: n_points={args.n_points}, dim={args.dim}")
    print(f"Measures compared: {'TSI + ' if args.with_tsi else ''}{len(baseline_results)} baselines")


if __name__ == "__main__":
    main()