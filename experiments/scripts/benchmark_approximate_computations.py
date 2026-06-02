#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.baselines import run_baseline_measures, run_approximate_baseline_measures
from src.tsi import EfficientTSI, BatchedTSI, ApproxTSI
from src.qsi import EfficientQSI, BatchedQSI, ApproxQSI
from src.data import RepresentationPair


def load_cifar10_representations(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10 initial and final epoch representations."""
    initial_path = data_dir / "cifar-10-initial-epoch-val-representations.npy"
    final_path = data_dir / "cifar-10-final-epoch-val-representations.npy"
    
    X = np.load(initial_path)
    Y = np.load(final_path)

    return X, Y


def compute_exact_scores(X: np.ndarray, Y: np.ndarray) -> dict:
    print("[Exact] Computing baseline measures...")
    # Baselines (exact)
    baseline_scores = run_baseline_measures(X, Y, time_monitor=False)

    # TSI/QSI (exact efficient versions)
    d = lambda a, b: np.linalg.norm(a - b)
    representations = RepresentationPair(X=X, Y=Y, d_x=d, d_y=d)

    print("[Exact] Computing TSI (efficient)...")
    tsi_score = EfficientTSI(euclidean=True, memory_efficient=False)(representations)
    print(f"[Exact] TSI done: {tsi_score:.6f}")
    print("[Exact] Computing QSI (efficient)...")
    qsi_score = EfficientQSI(euclidean=True)(representations)
    print(f"[Exact] QSI done: {qsi_score:.6f}")

    result = {
        'TSI': tsi_score,
        'QSI': qsi_score,
    }
    # baseline_scores keys are friendly names like 'CKA', 'CKNNA', ...
    result.update(baseline_scores)
    return result


def compute_approx_scores(X: np.ndarray, Y: np.ndarray, run_seed: int,
                         batch_size: int, no_batches: int,
                         n_threads: int) -> dict:
    print(f"[Approx run seed={run_seed}] Computing approximate baselines (batch_size={batch_size}, no_batches={no_batches})...")
    # Approximate baselines (mini-batch sampling)
    approx_baselines = run_approximate_baseline_measures(
        X, Y, batch_size=batch_size, no_batches=no_batches, seed=run_seed
    )

    # Approximate TSI/QSI (batch-based efficient variants only)
    d = lambda a, b: np.linalg.norm(a - b)
    representations = RepresentationPair(X=X, Y=Y, d_x=d, d_y=d)

    print(f"[Approx run seed={run_seed}] Computing BatchedTSI (batch_size={batch_size}, no_batches={no_batches})...")
    approx_tsi_efficient = BatchedTSI(euclidean=True, memory_efficient=False, n_threads=n_threads,
                                             batch_size=batch_size, no_batches=no_batches, seed=run_seed)(representations)
    print(f"[Approx run seed={run_seed}] BatchedTSI done: {approx_tsi_efficient:.6f}")

    print(f"[Approx run seed={run_seed}] Computing BatchedQSI (batch_size={batch_size}, no_batches={no_batches})...")
    approx_qsi_efficient = BatchedQSI(euclidean=True, n_threads=n_threads,
                                             batch_size=batch_size, no_batches=no_batches, seed=run_seed)(representations)
    print(f"[Approx run seed={run_seed}] BatchedQSI done: {approx_qsi_efficient:.6f}")

    approx_n_samples = (batch_size ** 2) * no_batches
    print(f"[Approx run seed={run_seed}] Computing ApproxTSI/QSI with n_samples={approx_n_samples}...")
    approx_tsi_sampling = ApproxTSI(n_samples=approx_n_samples, n_threads=n_threads, seed=run_seed)(representations)
    print(f"[Approx run seed={run_seed}] ApproxTSI(n_samples={approx_n_samples}) done: {approx_tsi_sampling:.6f}")
    approx_qsi_sampling = ApproxQSI(n_samples=approx_n_samples, n_threads=n_threads, seed=run_seed)(representations)
    print(f"[Approx run seed={run_seed}] ApproxQSI(n_samples={approx_n_samples}) done: {approx_qsi_sampling:.6f}")

    result = {
        'B-TSI': approx_tsi_efficient,
        'B-QSI': approx_qsi_efficient,
        'C-TSI': approx_tsi_sampling,
        'C-QSI': approx_qsi_sampling,
    }
    result.update(approx_baselines)
    return result


def main():
    parser = argparse.ArgumentParser(description='Benchmark approximate computations on CIFAR-10 initial vs final epoch representations.')
    parser.add_argument('--seed', type=int, default=0, help='Base RNG seed')

    parser.add_argument('--approx-runs', type=int, default=20, help='Number of approximation runs')
    parser.add_argument('--batch-size', type=int, default=1000, help='Mini-batch size for approximate methods')
    parser.add_argument('--no-batches-sweep', type=int, default=[1, 2, 4, 8, 16, 32, 64, 128],
                        help='Values of no_batches to sweep over for approximations')
    parser.add_argument('--n-threads', type=int, default=8, help='Threads for parallel approximate methods')

    args = parser.parse_args()

    # Load CIFAR-10 representations
    data_dir = Path(__file__).parent.parent.parent / 'data'
    print("[Setup] Loading CIFAR-10 representations...")
    X, Y = load_cifar10_representations(data_dir)
    
    # Derive n and dim from the loaded data
    n = X.shape[0]
    dim = X.shape[1]

    print("=== Benchmark: Approximate Computations (CIFAR-10 Initial vs Final Epoch) ===")
    print(f"n={n}, dim={dim}, seed={args.seed}")
    print(f"approx_runs={args.approx_runs}, batch_size={args.batch_size}, no_batches_sweep={args.no_batches_sweep}")
    print(f"n_threads={args.n_threads}")
    print(f"[Setup] Data ready: X.shape={X.shape} (initial epoch), Y.shape={Y.shape} (final epoch)")

    # Exact scores row
    print("[Exact] Starting exact computations...")
    exact_scores = compute_exact_scores(X, Y)
    print("[Exact] Baselines available:", ', '.join(sorted([k for k in exact_scores.keys() if k not in ('TSI','QSI')])))

    # Flatten exact into a dict of scalars (baseline runner returns scalars)
    exact_row = {
        'type': 'exact',
        'n': n,
        'dim': dim,
        'seed': args.seed,
    }
    # Baselines may include None for failures; store as NaN
    for k, v in exact_scores.items():
        if isinstance(v, tuple):
            # Safety: time_monitor=False so tuples should not occur, but handle anyway
            exact_row[k] = v[0]
        else:
            exact_row[k] = np.nan if v is None else v

    exact_df = pd.DataFrame([exact_row])

    # Approximation runs
    approx_rows = []
    for i in range(args.approx_runs):
        run_seed = args.seed + 1000 + i
        print(f"\n[Approx] Run {i+1}/{args.approx_runs} (seed={run_seed})")
        for nb in args.no_batches_sweep:
            print(f"[Approx] no_batches={nb}")
            approx_scores = compute_approx_scores(
                X, Y,
                run_seed=run_seed,
                batch_size=args.batch_size,
                no_batches=nb,
                n_threads=args.n_threads,
            )
            row = {
                'type': 'approx',
                'run': i + 1,
                'n': n,
                'dim': dim,
                'seed': run_seed,
                'no_batches': nb,
            }
            for k, v in approx_scores.items():
                row[k] = np.nan if v is None else v
            approx_rows.append(row)
        print(f"[Approx] Run {i+1} completed.")

    approx_df = pd.DataFrame(approx_rows)

    # Save
    results_dir = Path(__file__).parent.parent / 'results' / 'benchmark_approximate'
    results_dir.mkdir(parents=True, exist_ok=True)

    exact_path = results_dir / f"exact_scores_n{n}_d{dim}_seed{args.seed}.csv"
    approx_path = results_dir / f"approx_scores_n{n}_d{dim}_runs{args.approx_runs}_seed{args.seed}_batch{args.batch_size}.csv"

    exact_df.to_csv(exact_path, index=False)
    approx_df.to_csv(approx_path, index=False)

    print(f"\n[Output] Saved exact scores to: {exact_path}")
    print(f"[Output] Saved approximate scores to: {approx_path}")


if __name__ == '__main__':
    main()


