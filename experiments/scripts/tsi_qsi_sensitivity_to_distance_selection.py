#!/usr/bin/env python3
"""
TSI / QSI sensitivity to distance function choice.

Interpolates a second representation between independent noise and a copy of X,
then measures TSI and QSI between X and the interpolated representation for each
distance (Euclidean, negative normalized dot product, Manhattan).
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import RepresentationPair
from src.qsi import EfficientQSI
from src.tsi import EfficientTSI


def generate_random_data(n_points: int, dim: int, seed: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Generate two independent random datasets X and Y in [0, 1)."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_points, dim))
    rng_y = np.random.default_rng(seed + 1000 if seed is not None else None)
    Y = rng_y.random((n_points, dim))
    return X, Y


def blend_toward_signal(signal: np.ndarray, noise: np.ndarray, t: float) -> np.ndarray:
    """Linear blend: t=0 -> noise, t=1 -> signal."""
    return (1.0 - t) * noise + t * signal


def resolve_distance(name: str) -> tuple[str, Callable[[np.ndarray, np.ndarray], float]]:
    if name == "euclidean":
        return name, lambda x, y: float(np.linalg.norm(x - y, ord=2))
    if name == "cosine":

        def neg_norm_dot(x: np.ndarray, y: np.ndarray) -> float:
            nx = np.linalg.norm(x)
            ny = np.linalg.norm(y)
            denom = nx * ny
            if denom < 1e-12:
                return 0.0
            return float(-np.dot(x, y) / denom)

        return name, neg_norm_dot
    if name == "manhattan":
        return name, lambda x, y: float(np.linalg.norm(x - y, ord=1))
    raise ValueError(f"Unknown distance: {name}")


def benchmark_tsi(
    X: np.ndarray,
    Y: np.ndarray,
    d_func: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[float, float]:
    representations = RepresentationPair(X, Y, d_func, d_func)
    efficient_tsi = EfficientTSI()
    start_time = time.perf_counter()
    score = efficient_tsi(representations)
    elapsed = time.perf_counter() - start_time
    return float(score), elapsed


def benchmark_qsi(
    X: np.ndarray,
    Y: np.ndarray,
    d_func: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[float, float]:
    representations = RepresentationPair(X, Y, d_func, d_func)
    efficient_qsi = EfficientQSI()
    start_time = time.perf_counter()
    score = efficient_qsi(representations)
    elapsed = time.perf_counter() - start_time
    return float(score), elapsed


def run_experiment(args: argparse.Namespace) -> list[dict]:
    t_ticks = np.linspace(0.0, 1.0, args.time_ticks)
    results: list[dict] = []

    print(f"=== TSI / QSI sensitivity to distance selection ===")
    print(f"N={args.n}, D={args.d}, blend ticks={args.time_ticks}")
    print(f"Distance functions: {args.distance_functions}")
    print(f"Runs per (distance, tick): {args.runs}")
    print()

    for d_name_in in args.distance_functions:
        d_name, d_func = resolve_distance(d_name_in)
        print(f"--- Distance: {d_name} ---")

        for run_idx in range(args.runs):
            run_seed = args.seed + run_idx if args.seed is not None else None
            X, noise = generate_random_data(args.n, args.d, run_seed)
            print(f"  Run {run_idx + 1}/{args.runs} (seed={run_seed})")

            for t in t_ticks:
                noisy_X = blend_toward_signal(X, noise, float(t))
                row: dict = {
                    "distance_function": d_name,
                    "time_tick": float(t),
                    "run": run_idx + 1,
                    "seed": run_seed,
                    "n": args.n,
                    "d": args.d,
                }

                try:
                    tsi_score, tsi_time = benchmark_tsi(X, noisy_X, d_func)
                    row["tsi_score"] = tsi_score
                    row["tsi_time"] = tsi_time
                except Exception as e:
                    print(f"    t={t:.4f}: TSI failed: {e}")
                    row["tsi_score"] = np.nan
                    row["tsi_time"] = np.nan

                try:
                    qsi_score, qsi_time = benchmark_qsi(X, noisy_X, d_func)
                    row["qsi_score"] = qsi_score
                    row["qsi_time"] = qsi_time
                except Exception as e:
                    print(f"    t={t:.4f}: QSI failed: {e}")
                    row["qsi_score"] = np.nan
                    row["qsi_time"] = np.nan

                print(
                    f"    t={float(t):.4f}: TSI score={row.get('tsi_score', float('nan')):.6f}, "
                    f"time={row.get('tsi_time', float('nan')):.4f}s | "
                    f"QSI score={row.get('qsi_score', float('nan')):.6f}, "
                    f"time={row.get('qsi_time', float('nan')):.4f}s"
                )
                results.append(row)
            print()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TSI and QSI sensitivity to distance function selection.",
    )
    parser.add_argument(
        "--distance-functions",
        nargs="+",
        default=["euclidean", "cosine", "manhattan"],
        choices=["euclidean", "cosine", "manhattan"],
        metavar="DIST",
        help="Distance functions to evaluate (same metric on X and Y).",
    )
    parser.add_argument(
        "--time-ticks",
        type=int,
        default=10,
        dest="time_ticks",
        help="Number of blend values t in [0, 1].",
    )
    parser.add_argument("--d", type=int, default=50, help="Embedding dimensionality")
    parser.add_argument("--n", type=int, default=1000, help="Number of points per representation")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    parser.add_argument("--runs", type=int, default=1, help="Number of random runs per distance") # TODO: change to 20
    args = parser.parse_args()

    print("TSI / QSI sensitivity to distance selection")
    print(f"RNG seed (base): {args.seed}")
    print(f"Runs per distance: {args.runs}")
    print()

    all_results = run_experiment(args)

    df = pd.DataFrame(all_results)
    results_dir = Path(__file__).parent.parent / "results" / "tsi_qsi_sensitivity_to_distance_selection"
    results_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"tsi_qsi_sensitivity_to_distance_selection_d{args.d}_n{args.n}_"
        f"ticks{args.time_ticks}_runs{args.runs}_seed{args.seed}.csv"
    )
    filepath = results_dir / filename
    df.to_csv(filepath, index=False)
    print(f"Results saved to: {filepath}")

    print("\n=== SUMMARY (mean over runs) ===")
    if not df.empty and "tsi_score" in df.columns:
        summary = (
            df.groupby(["distance_function", "time_tick"])[["tsi_score", "qsi_score", "tsi_time", "qsi_time"]]
            .mean()
            .reset_index()
        )
        print(summary.to_string(index=False, float_format="%.6f"))
    print(f"\nTotal measurements: {len(all_results)}")


if __name__ == "__main__":
    main()
