import os
import csv
import time
import argparse
from find_all_representative_ordinal_Xs import get_all_representative_valid_Xs
import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from src.data import RepresentationPair
from src.tsi import TSI
from src.qsi import QSI


def absolute_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute absolute distance between two arrays (picklable for multiprocessing)."""
    return np.abs(x - y)


def create_metric_instance(metric_name: str):
    """Create a metric instance from its name (picklable for multiprocessing)."""
    if metric_name == "tsi":
        return TSI()
    elif metric_name == "qsi":
        return QSI()
    else:
        raise ValueError(f"Invalid metric: {metric_name}")

def compute_min_score_for_x1(
    idx1: int,
    valid_Xs: list,
    n: int,
    metric_name: str,
) -> tuple[int, float, np.ndarray]:
    """
    Compute the minimum score for X_1 against all subsequent X_2s.
    
    Args:
        idx1: Index of X_1 in valid_Xs
        valid_Xs: List of all valid X configurations
        n: Number of points
        metric_name: Name of the metric ("tsi" or "qsi")
    
    Returns:
        Tuple of (idx1, minimum score, X_1)
    """
    X_1 = valid_Xs[idx1]
    metric_instance = create_metric_instance(metric_name)
    min_score = 1.0
    
    for X_2 in valid_Xs[idx1 + 1:]:
        for permutation in itertools.permutations(range(n)):
            X_2_permuted = X_2[np.array(permutation)]
            representation_pair = RepresentationPair(
                np.array([X_1]).reshape(-1, 1),
                np.array([X_2_permuted]).reshape(-1, 1),
                absolute_distance,
                absolute_distance,
            )
            score = metric_instance(representation_pair)
            if score == 0:
                return (idx1, 0.0, X_1)
            elif score < min_score:
                min_score = score
    
    return (idx1, min_score, X_1)


def find_global_lower_bound(
    n: int,
    metric: str = "tsi",
    eps: float = 1e-6,
    use_networkx: bool = True,
    max_workers: int = 1,
) -> tuple[float, int, float]:
    """
    Find the global lower bound for the given metric.
    
    Args:
        n: Number of points
        metric: Metric name ("tsi" or "qsi")
        eps: Epsilon for numerical precision
        use_networkx: Whether to use networkx for topological sorting
        max_workers: Maximum number of parallel workers
    
    Returns:
        Tuple of (lowest score, number of configurations, elapsed time)
    """
    if metric == "qsi" and n < 4:
        raise ValueError("QSI requires n >= 4 points.")
    if metric == "tsi" and n < 3:
        raise ValueError("TSI requires n >= 3 points.")

    start_time = time.time()
    valid_Xs = get_all_representative_valid_Xs(
        n, eps=eps, use_networkx=use_networkx, metric=metric, max_workers=max_workers
    )
    num_configs = len(valid_Xs)
    print(f"Found {num_configs} valid X configurations for n={n}, metric={metric}")
    
    if num_configs <= 1:
        return 1.0, num_configs, time.time() - start_time
    
    # Convert to list of numpy arrays for consistent handling
    valid_Xs_arrays = [
        np.array(x) if not isinstance(x, np.ndarray) else x for x in valid_Xs
    ]
    
    # Create partial function with fixed arguments
    compute_fn = partial(
        compute_min_score_for_x1,
        valid_Xs=valid_Xs_arrays,
        n=n,
        metric_name=metric,
    )
    
    # Process all X_1 indices in parallel
    indices = range(len(valid_Xs_arrays) - 1)  # -1 because last X has no subsequent X
    num_tasks = len(indices)
    lowest_score = 1.0
    
    # Work estimation: Task i compares against (len - 1 - i) subsequent configs
    total_work = sum(len(valid_Xs_arrays) - 1 - i for i in indices)
    work_done = 0
    completed = 0
    
    print(f"Starting global lower bound search for {num_tasks} task groups...")
    lb_start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(compute_fn, i): i for i in indices}
        for future in as_completed(future_to_index):
            idx1 = future_to_index[future]
            try:
                _, min_score, X_1 = future.result()
                completed += 1
                
                # Update work done
                work_done += (len(valid_Xs_arrays) - 1 - idx1)
                

                min_score = float(min_score.item() if hasattr(min_score, "item") else min_score)

                if min_score < lowest_score:
                    lowest_score = min_score
                
                elapsed = time.time() - lb_start_time
                if work_done > 0:
                    avg_time_per_work = elapsed / work_done
                    remaining_work = total_work - work_done
                    est_time_left = avg_time_per_work * remaining_work
                else:
                    est_time_left = 0
                
                print(f"\rLB Search: {completed}/{num_tasks} groups. Lowest: {float(lowest_score):.4f}. "
                      f"Elapsed: {elapsed:.1f}s, Est. left: {est_time_left:.1f}s", end="", flush=True)
                
                if lowest_score == 0:
                    print("\nFound score 0, stopping early.")
                    executor.shutdown(wait=False, cancel_futures=True)
                    return 0.0, num_configs, time.time() - start_time
                    
            except Exception as exc:
                print(f"\nIndex {idx1} generated an exception: {exc}")

    total_time = time.time() - start_time
    print(f"\nCompleted n={n}, metric={metric} in {total_time:.1f}s. Global LB: {lowest_score}")
    return lowest_score, num_configs, total_time


def compute_bounds_and_save(
    ns: list[int],
    metric: str,
    eps: float = 1e-6,
    use_networkx: bool = True,
    max_workers: int = 1,
    output_file: str = None,
):
    """
    Compute global lower bounds for multiple n values and a single metric, saving results to CSV.
    
    Args:
        ns: List of n values to compute bounds for
        metric: Metric name ("tsi" or "qsi")
        eps: Epsilon for numerical precision
        use_networkx: Whether to use networkx for topological sorting
        max_workers: Maximum number of parallel workers
        output_file: Output file path (if None, will be auto-generated based on metric)
    """
    # Generate output file name based on metric if not provided
    if output_file is None:
        output_file = f"proofs/global_lower_bounds_{metric}.csv"
    
    # Header for the CSV
    header = ["n", "metric", "lower_bound", "num_configs", "eps", "time_seconds"]
    
    for n in ns:
        # Check for minimum n requirements per metric
        if metric == "qsi" and n < 4:
            print(f"Skipping n={n} for qsi (minimum n=4 required)")
            continue
        if metric == "tsi" and n < 3:
            print(f"Skipping n={n} for tsi (minimum n=3 required)")
            continue

        print(f"\n{'='*50}")
        print(f"Computing Global LB for n={n}, metric={metric}")
        print(f"{'='*50}")
        
        lower_bound, num_configs, elapsed_time = find_global_lower_bound(
            n, metric, eps, use_networkx, max_workers
        )
        
        # Save/Update result immediately
        mode = 'a' if os.path.isfile(output_file) else 'w'
        with open(output_file, mode, newline='') as f:
            writer = csv.writer(f)
            if mode == 'w':
                writer.writerow(header)
            writer.writerow([n, metric, lower_bound, num_configs, eps, f"{elapsed_time:.2f}"])
        
        print(f"Saved result for n={n}, metric={metric} to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute global lower bounds for ordinal similarity metrics.'
    )
    parser.add_argument(
        '--metric',
        type=str,
        required=True,
        choices=['tsi', 'qsi'],
        help='Metric to compute bounds for: "tsi" or "qsi"'
    )
    parser.add_argument(
        '--ns',
        type=int,
        nargs='+',
        default=[3, 4, 5, 6],
        help='List of n values to compute bounds for (default: [3, 4, 5, 6])'
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-3,
        help='Epsilon for numerical precision (default: 1e-3)'
    )
    parser.add_argument(
        '--use-networkx',
        action='store_true',
        default=True,
        help='Use networkx for topological sorting (default: True)'
    )
    parser.add_argument(
        '--no-networkx',
        dest='use_networkx',
        action='store_false',
        help='Disable networkx for topological sorting'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Maximum number of parallel workers (default: number of CPUs)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output CSV file path (default: proofs/global_lower_bounds_{metric}.csv)'
    )

    args = parser.parse_args()
    
    # Set max_workers to CPU count if not specified
    max_workers = args.max_workers if args.max_workers is not None else os.cpu_count()
    
    print(f"Computing global lower bounds for metric: {args.metric}")
    print(f"n values: {args.ns}")
    print(f"eps: {args.eps}")
    print(f"use_networkx: {args.use_networkx}")
    print(f"max_workers: {max_workers}")
    
    output_file = args.output_file
    if output_file is None:
        output_file = f"proofs/global_lower_bounds_{args.metric}.csv"
    print(f"Output file: {output_file}")
    
    compute_bounds_and_save(
        ns=args.ns,
        metric=args.metric,
        eps=args.eps,
        use_networkx=args.use_networkx,
        max_workers=max_workers,
        output_file=output_file
    )