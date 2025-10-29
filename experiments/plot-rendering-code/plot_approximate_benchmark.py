#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

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


def load_results(results_dir: Path, n: int, dim: int, seed: int, approx_runs: int | None) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    exact_path = results_dir / f"exact_scores_n{n}_d{dim}_seed{seed}.csv"
    if approx_runs is None:
        # If approx_runs not provided, try to infer by globbing
        candidates = sorted(results_dir.glob(f"approx_scores_n{n}_d{dim}_runs*_seed{seed}.csv"))
        if not candidates:
            raise FileNotFoundError("Approx results file not found; specify --approx-runs or ensure file exists.")
        approx_path = candidates[-1]
    else:
        approx_path = results_dir / f"approx_scores_n{n}_d{dim}_runs{approx_runs}_seed{seed}.csv"

    if not exact_path.exists():
        raise FileNotFoundError(f"Missing exact results: {exact_path}")
    if not approx_path.exists():
        raise FileNotFoundError(f"Missing approx results: {approx_path}")

    exact_df = pd.read_csv(exact_path)
    approx_df = pd.read_csv(approx_path)
    return exact_df, approx_df, exact_path, approx_path


def main():
    parser = argparse.ArgumentParser(description="Plot approximate benchmark results vs exact")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--approx-runs", type=int, default=None, help="Approx runs used when saving file (optional)")
    parser.add_argument("--metric", type=str, default="TSI", help="Exact metric column to compare (e.g., TSI, QSI)")
    parser.add_argument(
        "--approx-metric",
        type=str,
        default=None,
        help="Approximate metric column to use (e.g., EfficientApproxTSI). If omitted, inferred from --metric.")
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

    exact_df, approx_df, exact_path, approx_path = load_results(args.results_dir, args.n, args.dim, args.seed, args.approx_runs)

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
            g = (
                approx_df[["no_batches", col]]
                .assign(deviation=lambda d, ex=exact_value, ac=col: (d[ac] - ex).abs())
                .groupby("no_batches")
                .agg(mean_dev=("deviation", "mean"), std_dev=("deviation", "std"))
                .reset_index()
                .sort_values("no_batches")
            )
            grouped_per_variant[(measure, variant)] = g

    # X axis values (assume same set for all)
    x_no_batches = np.sort(approx_df["no_batches"].unique())
    # Final estimate stats at max no_batches for each (measure, variant)
    max_nb = int(x_no_batches.max())
    # final_stats structure: measure -> { 'exact': val, 'B': (mean,std)?, 'C': (mean,std)? }
    final_stats: dict[str, dict[str, float | tuple[float, float]]] = {}
    final_subset = approx_df[approx_df["no_batches"] == max_nb]
    for measure in sorted(measure_names):
        entry: dict[str, float | tuple[float, float]] = {"exact": float(exact_row[measure])}
        if "B" in measure_to_variants[measure]:
            col = measure_to_variants[measure]["B"]
            m = float(final_subset[col].mean())
            s = float(final_subset[col].std(ddof=1)) if len(final_subset) > 1 else 0.0
            entry["B"] = (m, s)
        if "C" in measure_to_variants[measure]:
            col = measure_to_variants[measure]["C"]
            m = float(final_subset[col].mean())
            s = float(final_subset[col].std(ddof=1)) if len(final_subset) > 1 else 0.0
            entry["C"] = (m, s)
        final_stats[measure] = entry

    # Plot (match sizing aesthetics)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(23, 8))

    # Colors: use a different scheme for the first plot
    colors = plt.cm.Dark2.colors

    # Tick formatter similar to other scripts
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

    # Subplot 1: deviation vs no_batches for all metric variants
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    from matplotlib.lines import Line2D
    legend_handles = []
    
    # First pass: plot lines and error bars (no markers)
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

    for i, ((measure, variant), g) in enumerate(ordered_variant_items):
        label = f"{variant}-{measure}"
        mdev = g['mean_dev'].to_numpy()
        sdev = g['std_dev'].fillna(0.0).to_numpy()
        # Avoid zeros on log scale
        eps = 1e-12
        mdev = np.clip(mdev, eps, None)
        ax1.plot(g['no_batches'], mdev,
                 linestyle=linestyles[i % len(linestyles)],
                 linewidth=5.6,
                 color=colors[i % len(colors)],
                 alpha=0.9, zorder=2)
        # Removed standard deviation error bars per request
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
        sdev = g['std_dev'].fillna(0.0).to_numpy()
        # Avoid zeros on log scale
        eps = 1e-12
        mdev = np.clip(mdev, eps, None)
        marker_offset = i % 3
        ax1.plot(g['no_batches'], mdev,
                 marker=markers[i % len(markers)],
                 linestyle='',
                 markersize=28,
                 color=colors[i % len(colors)],
                 markeredgewidth=2.8, markeredgecolor='white',
                 markevery=(marker_offset, 3), zorder=10)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    nb_ticks = sorted(list(x_no_batches))
    ax1.set_xticks(nb_ticks)
    ax1.xaxis.set_major_formatter(formatter)
    ax1.set_xlabel('Number of Batches', fontsize=30, fontweight='bold')
    ax1.set_ylabel('Absolute Deviation', fontsize=30, fontweight='bold')
    ax1.set_ylim(1e-4, 0.5)
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=4)
    ax1.set_yticks([5e-1, 1e-1, 1e-2, 1e-3, 1e-4])
    ax1.set_yticklabels(['', '1e-1', '1e-2', '1e-3', '1e-4'])
    leg = ax1.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.01, 0.5),
                    borderaxespad=0.0, fontsize=28, framealpha=0.95, edgecolor='black',
                    fancybox=False, shadow=False, ncol=1)
    # Increase legend handle sizes to match other plots
    for lh in leg.legend_handles:
        lh.set_linewidth(6.4)
        lh.set_markersize(28)

    # Subplot 2: final estimate vs exact (grouped barplot with optional custom)
    # Ordered metric names for bars
    requested_order = ["TSI", "QSI", "CKA", "CKNNA", "MutualNN", "SVCCA", "PWCCA"]
    metric_names = [m for m in requested_order if m in final_stats]
    x = np.arange(len(metric_names))

    width = 0.25
    exact_vals = [final_stats[m]["exact"] for m in metric_names]
    ax2.bar(x - width, exact_vals, width, label='Exact', color=colors[0], alpha=0.9)

    # B-Approx bars
    b_means = []
    b_stds = []
    for m in metric_names:
        if "B" in final_stats[m]:
            mean_b, std_b = final_stats[m]["B"]  # type: ignore[index]
        else:
            mean_b, std_b = np.nan, 0.0
        b_means.append(mean_b)
        b_stds.append(std_b)
    ax2.bar(x, b_means, width, label=f'Batched Approx @ {max_nb}', color=colors[1], alpha=0.9)

    # C-Approx bars (only where present)
    c_means = []
    c_stds = []
    for m in metric_names:
        if "C" in final_stats[m]:
            mean_c, std_c = final_stats[m]["C"]  # type: ignore[index]
        else:
            mean_c, std_c = np.nan, 0.0
        c_means.append(mean_c)
        c_stds.append(std_c)
    ax2.bar(x + width, c_means, width, label=f'Custom Approx @ {max_nb}', color=colors[2] if len(colors) > 2 else 'gray', alpha=0.9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_names, rotation=30, ha='right')
    ax2.set_ylabel('Similarity Score', fontsize=30, fontweight='bold')
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, axis='y', alpha=0.2, linestyle='--', linewidth=4)
    ax2.legend(fontsize=24, framealpha=0.95, edgecolor='black')

    # Keep a small right margin so the legend hugs the plots without wasting space
    plt.tight_layout(rect=[0, 0, 0.95, 1])

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


