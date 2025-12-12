#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

# Style consistent with other plots
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 28
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28
plt.rcParams['legend.fontsize'] = 32
plt.rcParams['figure.titlesize'] = 30


def find_column(df: pd.DataFrame, preferred: str, fallbacks: list[str]) -> str:
    if preferred in df.columns:
        return preferred
    for fb in fallbacks:
        if fb in df.columns:
            return fb
    # Try substring match as a last resort
    for col in df.columns:
        if preferred.lower() in col.lower():
            return col
    raise KeyError(f"None of the columns found. Preferred: {preferred}, fallbacks: {fallbacks}")


def load_results(results_dir: Path, n: int, dim: int, seed: int, approx_runs: int | None, batch_size: int) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """Load exact and approx results for a specific batch_size."""
    exact_path = results_dir / f"exact_scores_n{n}_d{dim}_seed{seed}.csv"
    if approx_runs is None:
        # If approx_runs not provided, try to infer by globbing
        candidates = sorted(results_dir.glob(f"approx_scores_n{n}_d{dim}_runs*_seed{seed}_batch{batch_size}.csv"))
        if not candidates:
            raise FileNotFoundError(f"Approx results file not found for batch_size={batch_size}; specify --approx-runs or ensure file exists.")
        approx_path = candidates[-1]
    else:
        approx_path = results_dir / f"approx_scores_n{n}_d{dim}_runs{approx_runs}_seed{seed}_batch{batch_size}.csv"

    if not exact_path.exists():
        raise FileNotFoundError(f"Missing exact results: {exact_path}")
    if not approx_path.exists():
        raise FileNotFoundError(f"Missing approx results: {approx_path}")

    exact_df = pd.read_csv(exact_path)
    approx_df = pd.read_csv(approx_path)
    return exact_df, approx_df, exact_path, approx_path


def prepare_deviation_data(exact_df: pd.DataFrame, approx_df: pd.DataFrame) -> tuple[dict, np.ndarray, list[str]]:
    """
    Prepare deviation data for plotting.
    Returns:
        grouped_per_variant: dict mapping (measure, variant) -> DataFrame with mean_dev, std_dev per no_batches
        x_no_batches: sorted array of no_batches values
        measure_names: list of measure names that have both exact and approx counterparts
    """
    # Prepare comparable metric groups by measure and variant (B- or C-)
    reserved = {"type", "run", "n", "dim", "seed", "no_batches"}
    approx_cols = [c for c in approx_df.columns if c not in reserved]
    exact_cols = set([c for c in exact_df.columns if c not in reserved])

    # Build mapping: measure_name -> {variant -> column_name}
    # Variants: 'B' (Direct Batched), 'C' (Custom). Legacy names are mapped where possible.
    measure_to_variants: dict[str, dict[str, str]] = {}
    def _add_variant(measure: str, variant: str, col: str):
        if measure not in measure_to_variants:
            measure_to_variants[measure] = {}
        measure_to_variants[measure][variant] = col

    for c in approx_cols:
        if c.startswith("B-"):
            _add_variant(c[2:], "B", c)
        elif c.startswith("C-"):
            _add_variant(c[2:], "C", c)
        elif c == "AP-S-CKA":
            _add_variant("CKA", "C", c)
        elif c in exact_cols:
            # Legacy case where approx column equals exact name; treat as B variant
            _add_variant(c, "B", c)

    # Keep only measures that have an exact counterpart
    measure_names = [m for m in measure_to_variants.keys() if m in exact_cols]
    if not measure_names:
        raise RuntimeError("No comparable metrics found between exact and approx results.")

    # Extract scalar exact values for each metric (from the single-row exact_df)
    if len(exact_df) < 1:
        raise RuntimeError("Exact CSV has no rows.")
    exact_row = exact_df.iloc[0]

    # Prepare data grouped by no_batches per (measure, variant)
    if "no_batches" not in approx_df.columns:
        raise KeyError("Column 'no_batches' not found in approx results; script expects sweeps over no_batches")

    grouped_per_variant: dict[tuple[str, str], pd.DataFrame] = {}
    for measure in sorted(measure_names):
        exact_value = float(exact_row[measure])
        for variant, col in measure_to_variants[measure].items():
            # Check if the column has any valid (non-NaN) values
            if approx_df[col].isna().all():
                # Skip this variant if all values are NaN
                continue
            g = (
                approx_df[["no_batches", col]]
                .assign(deviation=lambda d, ex=exact_value, ac=col: (d[ac] - ex).abs())
                .groupby("no_batches")
                .agg(mean_dev=("deviation", "mean"), std_dev=("deviation", "std"))
                .reset_index()
                .sort_values("no_batches")
            )
            # Also skip if all deviations are NaN after grouping
            if g['mean_dev'].isna().all():
                continue
            grouped_per_variant[(measure, variant)] = g

    # X axis values (assume same set for all)
    x_no_batches = np.sort(approx_df["no_batches"].unique())
    
    return grouped_per_variant, x_no_batches, measure_names


def plot_deviation_subplot(ax, grouped_per_variant: dict, x_no_batches: np.ndarray, 
                           colors, markers, linestyles, formatter,
                           batch_size: int, show_ylabel: bool = True):
    """Plot deviation vs no_batches on a single subplot."""
    legend_handles = []
    
    # Enforce requested ordering for measures and variants
    requested_order = ["TSI", "QSI", "CKA", "CKNNA", "MutualNN", "SVCCA", "PWCCA"]
    all_keys = set(grouped_per_variant.keys())
    ordered_variant_items = []
    for m in requested_order:
        for v in ["B", "C"]:
            key = (m, v)
            if key in all_keys:
                ordered_variant_items.append((key, grouped_per_variant[key]))
    # Fallback: include any remaining in stable order
    for key, g in grouped_per_variant.items():
        if key not in dict(ordered_variant_items):
            ordered_variant_items.append((key, g))

    # First pass: plot lines (no markers)
    for i, ((measure, variant), g) in enumerate(ordered_variant_items):
        label = f"{variant}-{measure}"
        mdev = g['mean_dev'].to_numpy()
        # Avoid zeros on log scale
        eps = 1e-12
        mdev = np.clip(mdev, eps, None)
        ax.plot(g['no_batches'], mdev,
                linestyle=linestyles[i % len(linestyles)],
                linewidth=5.6,
                color=colors[i % len(colors)],
                alpha=0.9, zorder=2)
        legend_handles.append(Line2D([0], [0],
                                     color=colors[i % len(colors)],
                                     linestyle=linestyles[i % len(linestyles)],
                                     linewidth=2.5,
                                     marker=markers[i % len(markers)],
                                     markersize=7,
                                     markeredgewidth=1.5,
                                     markeredgecolor='white',
                                     label=label))
    
    # Second pass: overlay markers on top
    for i, ((measure, variant), g) in enumerate(ordered_variant_items):
        mdev = g['mean_dev'].to_numpy()
        eps = 1e-12
        mdev = np.clip(mdev, eps, None)
        marker_offset = i % 3
        ax.plot(g['no_batches'], mdev,
                marker=markers[i % len(markers)],
                linestyle='',
                markersize=28,
                color=colors[i % len(colors)],
                markeredgewidth=2.8, markeredgecolor='white',
                markevery=(marker_offset, 3), zorder=10)
    
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    nb_ticks = sorted(list(x_no_batches))
    ax.set_xticks(nb_ticks)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel('Number of Batches (B)', fontsize=30, fontweight='bold')
    if show_ylabel:
        ax.set_ylabel('Absolute Deviation', fontsize=30, fontweight='bold')
    ax.set_ylim(1e-5, 5e-1)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=4)
    ax.set_yticks([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    ax.set_yticklabels(['1e-1', '1e-2', '1e-3', '1e-4', '1e-5'])
    # Remove minor ticks on y-axis
    ax.yaxis.set_minor_locator(plt.NullLocator())
    
    # Add batch size label in bottom left corner
    ax.text(0.03, 0.05, r'$\mathbf{N_B =}$' + f'{batch_size}', transform=ax.transAxes,
            fontsize=30, fontweight='bold', va='bottom', ha='left')
    
    return legend_handles


def main():
    parser = argparse.ArgumentParser(description="Plot approximate benchmark results vs exact for multiple batch sizes")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--approx-runs", type=int, default=None, help="Approx runs used when saving file (optional)")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs='+',
        default=[100, 1000],
        help="Batch sizes to compare (default: 100 1000)"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "benchmark_approximate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "plots" / "approximate_benchmark_combined.png",
    )

    args = parser.parse_args()
    
    batch_sizes = args.batch_sizes
    if len(batch_sizes) != 2:
        raise ValueError("Exactly 2 batch sizes are required for the side-by-side plot")

    # Load data for each batch size
    data_per_batch = {}
    for bs in batch_sizes:
        exact_df, approx_df, exact_path, approx_path = load_results(
            args.results_dir, args.n, args.dim, args.seed, args.approx_runs, bs
        )
        grouped, x_no_batches, measure_names = prepare_deviation_data(exact_df, approx_df)
        data_per_batch[bs] = {
            'grouped': grouped,
            'x_no_batches': x_no_batches,
            'measure_names': measure_names,
        }
        print(f"Loaded data for batch_size={bs}: {approx_path}")

    # Plot setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(23, 7), sharey=True)
    
    # Colors and styling
    colors = plt.cm.Dark2.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    
    # Tick formatter
    def compact_tick_formatter(value, _pos):
        try:
            v = float(value)
        except Exception:
            return str(value)
        if v >= 1000:
            k = v / 1000.0
            return f"{k:.1f}k".replace(".0k", "k")
        return f"{int(v)}" if v.is_integer() else f"{v:g}"
    formatter = FuncFormatter(compact_tick_formatter)

    # Plot first batch size (left subplot)
    bs1 = batch_sizes[0]
    legend_handles = plot_deviation_subplot(
        ax1,
        data_per_batch[bs1]['grouped'],
        data_per_batch[bs1]['x_no_batches'],
        colors, markers, linestyles, formatter,
        batch_size=bs1,
        show_ylabel=True
    )
    
    # Plot second batch size (right subplot)
    bs2 = batch_sizes[1]
    plot_deviation_subplot(
        ax2,
        data_per_batch[bs2]['grouped'],
        data_per_batch[bs2]['x_no_batches'],
        colors, markers, linestyles, formatter,
        batch_size=bs2,
        show_ylabel=False
    )
    
    # Add shared legend to the right of the plots
    leg = ax2.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.01, 0.5),
                     borderaxespad=0.0, fontsize=28, framealpha=0.95, edgecolor='black',
                     fancybox=False, shadow=False, ncol=1)
    # Increase legend handle sizes to match other plots
    for lh in leg.legend_handles:
        lh.set_linewidth(6.4)
        lh.set_markersize(28)

    # Adjust layout
    plt.tight_layout()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
