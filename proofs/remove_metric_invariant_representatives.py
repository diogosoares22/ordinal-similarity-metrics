import os
import argparse
from find_all_representative_ordinal_Xs import (
    load_representative_Xs,
    remove_all_metric_invariant_Xs,
    get_representative_set_filepath,
    REPRESENTATIVE_SETS_DIR,
)
import numpy as np


# Default directory for storing metric-filtered representative sets
METRIC_FILTERED_SETS_DIR = "proofs/metric_invariant_representative_sets"


def get_available_cpus() -> int:
    """Get the number of CPUs available to this process (cross-platform)."""
    try:
        return len(os.sched_getaffinity(0))  # Linux (respects SLURM, cgroups, taskset)
    except AttributeError:
        return os.cpu_count()  # macOS, Windows fallback


def get_metric_filtered_filepath(n: int, eps: float, metric: str, output_dir: str = None) -> str:
    """
    Get the filepath for storing/loading metric-filtered representative sets.
    
    Args:
        n: Number of points
        eps: Epsilon for numerical precision
        metric: Metric name ("tsi" or "qsi")
        output_dir: Base output directory (defaults to METRIC_FILTERED_SETS_DIR)
    
    Returns:
        Filepath string
    """
    if output_dir is None:
        output_dir = METRIC_FILTERED_SETS_DIR
    
    # Metric-specific subdirectory
    metric_dir = os.path.join(output_dir, metric)
    
    # Format eps for filename (e.g., 1e-3 -> "1e-3", 0.001 -> "0.001")
    eps_str = f"{eps:.0e}" if eps < 0.01 else f"{eps}"
    return os.path.join(metric_dir, f"representative_Xs_n{n}_eps{eps_str}.npy")


def save_metric_filtered_Xs(
    Xs: list,
    n: int,
    eps: float,
    metric: str,
    output_dir: str = None,
) -> str:
    """
    Save metric-filtered representative X configurations to a file.
    
    Args:
        Xs: List of X configurations (numpy arrays)
        n: Number of points
        eps: Epsilon used for numerical precision
        metric: Metric name ("tsi" or "qsi")
        output_dir: Base output directory (defaults to METRIC_FILTERED_SETS_DIR)
    
    Returns:
        Path to the saved file
    """
    if output_dir is None:
        output_dir = METRIC_FILTERED_SETS_DIR
    
    # Create metric-specific subdirectory
    metric_dir = os.path.join(output_dir, metric)
    os.makedirs(metric_dir, exist_ok=True)
    
    filepath = get_metric_filtered_filepath(n, eps, metric, output_dir)
    
    # Convert to numpy array and save
    Xs_array = np.array([np.array(x) for x in Xs])
    np.save(filepath, Xs_array)
    
    print(f"Saved {len(Xs)} {metric}-filtered representative X configurations to {filepath}")
    return filepath


def load_metric_filtered_Xs(
    n: int,
    eps: float,
    metric: str,
    input_dir: str = None,
) -> list[np.ndarray] | None:
    """
    Load metric-filtered representative X configurations from a file.
    
    Args:
        n: Number of points
        eps: Epsilon used for numerical precision
        metric: Metric name ("tsi" or "qsi")
        input_dir: Base input directory (defaults to METRIC_FILTERED_SETS_DIR)
    
    Returns:
        List of X configurations as numpy arrays, or None if file doesn't exist
    """
    if input_dir is None:
        input_dir = METRIC_FILTERED_SETS_DIR
    
    filepath = get_metric_filtered_filepath(n, eps, metric, input_dir)
    
    if not os.path.exists(filepath):
        print(f"Metric-filtered representative set file not found: {filepath}")
        return None
    
    Xs_array = np.load(filepath)
    Xs_list = [Xs_array[i] for i in range(len(Xs_array))]
    
    print(f"Loaded {len(Xs_list)} {metric}-filtered representative X configurations from {filepath}")
    return Xs_list


def filter_and_save_representative_set(
    n: int,
    metric: str,
    eps: float = 1e-6,
    max_workers: int = 1,
    input_dir: str = None,
    output_dir: str = None,
) -> str | None:
    """
    Load representative Xs, apply metric-specific filtering, and save the result.
    
    Args:
        n: Number of points
        metric: Metric name ("tsi" or "qsi")
        eps: Epsilon for numerical precision
        max_workers: Maximum number of parallel workers
        input_dir: Directory containing base representative sets
        output_dir: Directory to save metric-filtered sets
    
    Returns:
        Path to the saved file, or None if input file not found
    """
    print(f"\n{'='*50}")
    print(f"Filtering representative set for n={n}, metric={metric}, eps={eps}")
    print(f"{'='*50}")
    
    # Load base representative Xs
    valid_Xs = load_representative_Xs(n, eps, input_dir)
    
    if valid_Xs is None:
        input_dir_display = input_dir or REPRESENTATIVE_SETS_DIR
        print(f"Error: Representative set not found for n={n}, eps={eps}.")
        print(f"Please run 'python find_all_representative_ordinal_Xs.py --ns {n} --eps {eps}' first "
              f"to generate the representative set in {input_dir_display}.")
        return None
    
    print(f"Loaded {len(valid_Xs)} base representative X configurations")
    
    # Apply metric-specific invariant removal
    print(f"Applying {metric}-specific invariant removal...")
    filtered_Xs = remove_all_metric_invariant_Xs(valid_Xs, metric, max_workers=max_workers)
    
    print(f"After filtering: {len(filtered_Xs)} configurations (removed {len(valid_Xs) - len(filtered_Xs)})")
    
    # Save the filtered result
    filepath = save_metric_filtered_Xs(filtered_Xs, n, eps, metric, output_dir)
    
    return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Remove metric-invariant representatives and save to metric-specific directories. '
                    'Requires pre-computed representative sets (run find_all_representative_ordinal_Xs.py first).'
    )
    parser.add_argument(
        '--metric',
        type=str,
        required=True,
        choices=['tsi', 'qsi'],
        help='Metric to filter for: "tsi" or "qsi"'
    )
    parser.add_argument(
        '--ns',
        type=int,
        nargs='+',
        default=[3, 4, 5, 6],
        help='List of n values to process (default: [3, 4, 5, 6])'
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-3,
        help='Epsilon for numerical precision (default: 1e-3)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Maximum number of parallel workers (default: number of CPUs)'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=None,
        help=f'Directory containing base representative sets (default: {REPRESENTATIVE_SETS_DIR})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help=f'Output directory for metric-filtered sets (default: {METRIC_FILTERED_SETS_DIR})'
    )

    args = parser.parse_args()
    
    max_workers = args.max_workers if args.max_workers is not None else get_available_cpus()
    input_dir = args.input_dir or REPRESENTATIVE_SETS_DIR
    output_dir = args.output_dir or METRIC_FILTERED_SETS_DIR
    
    print(f"Filtering representative sets for metric: {args.metric}")
    print(f"n values: {args.ns}")
    print(f"eps: {args.eps}")
    print(f"max_workers: {max_workers}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}/{args.metric}/")
    
    # Check for minimum n requirements per metric
    valid_ns = []
    for n in args.ns:
        if args.metric == "qsi" and n < 4:
            print(f"Skipping n={n} for qsi (minimum n=4 required)")
            continue
        if args.metric == "tsi" and n < 3:
            print(f"Skipping n={n} for tsi (minimum n=3 required)")
            continue
        valid_ns.append(n)
    
    for n in valid_ns:
        filter_and_save_representative_set(
            n=n,
            metric=args.metric,
            eps=args.eps,
            max_workers=max_workers,
            input_dir=input_dir,
            output_dir=output_dir,
        )
    
    print(f"\n{'='*50}")
    print("Metric-invariant filtering completed!")
    print(f"{'='*50}")

