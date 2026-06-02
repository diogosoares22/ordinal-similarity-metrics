#!/usr/bin/env python3
"""
Plot TSI / QSI sensitivity to distance function choice from CSV results.

Reads outputs of tsi_qsi_sensitivity_to_distance_selection.py (mean over runs per
distance and time tick). X-axis is the blend parameter ``time_tick`` (denoted
$t$ in the experiment: noise at 0, full signal at 1). X tick labels use
$0$, $k/(t-1)$ for $k=1,\ldots,t-2$, and $1$ when there are $t$ ticks (e.g. 1/9 ... 8/9 for $t=10$).
Y-axis is similarity score.

Produces two subplots: TSI (left) and QSI (right), each with one line per
distance (Euclidean, cosine / neg. normalized dot product, Manhattan).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# Style consistent with other experiment plots (e.g. plot_independent_benchmark.py)
plt.style.use("seaborn-v0_8-paper")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 28
plt.rcParams["axes.labelsize"] = 30
plt.rcParams["axes.titlesize"] = 30
plt.rcParams["xtick.labelsize"] = 28
plt.rcParams["ytick.labelsize"] = 28
plt.rcParams["legend.fontsize"] = 32
plt.rcParams["figure.titlesize"] = 30

DEFAULT_RESULTS_SUBDIR = "tsi_qsi_sensitivity_to_distance_selection"

# Match experiment script choices; stable line colors across subplots
DISTANCE_ORDER = ("euclidean", "cosine", "manhattan")
DISTANCE_LABELS = {
    "euclidean": "Euclidean",
    "cosine": "Cosine",
    "manhattan": "Manhattan",
}

Y_SCORE_TICKS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def make_blend_time_tick_formatter(t_ticks_sorted: list[float]) -> FuncFormatter:
    """
    Format blend-axis ticks as 0, 1/(t-1), 2/(t-1), …, (t-2)/(t-1), 1 for t tick
    values (same grid as ``np.linspace(0, 1, t)``).
    """
    ticks = sorted(float(x) for x in t_ticks_sorted)
    n = len(ticks)
    if n < 2:
        return FuncFormatter(lambda v, _p: f"{float(v):g}")

    denom = n - 1
    min_gap = min(ticks[i + 1] - ticks[i] for i in range(n - 1))
    match_tol = max(1e-12, 0.05 * min_gap)

    def fmt(value, _pos):
        try:
            x = float(value)
        except (TypeError, ValueError):
            return str(value)
        idx = min(range(n), key=lambda j: abs(ticks[j] - x))
        if abs(ticks[idx] - x) > match_tol:
            return f"{x:g}"
        if idx == 0:
            return "0"
        if idx == n - 1:
            return "1"
        return f"{idx}/{denom}"

    return FuncFormatter(fmt)


def apply_similarity_score_y_axis(ax) -> None:
    ax.set_yticks(Y_SCORE_TICKS)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{float(v):.1f}"))


def find_csv_files(results_dir: Path) -> list[Path]:
    csv_files = list(results_dir.glob("*.csv"))
    return sorted(csv_files)


def ordered_distances(distances: list[str]) -> list[str]:
    seen = set(distances)
    ordered = [d for d in DISTANCE_ORDER if d in seen]
    for d in sorted(seen):
        if d not in ordered:
            ordered.append(d)
    return ordered


def aggregate_by_distance_tick(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of scores over runs, per distance and time_tick."""
    group_cols = ["distance_function", "time_tick"]
    score_cols = [c for c in ("tsi_score", "qsi_score") if c in df.columns]
    if not score_cols:
        raise ValueError("CSV must contain tsi_score and/or qsi_score columns.")
    return df.groupby(group_cols, as_index=False)[score_cols].mean()


def infer_n_d(df: pd.DataFrame) -> tuple[int, int]:
    """Read N and D from experiment CSV (same convention as other benchmark plots)."""
    if "n" not in df.columns or "d" not in df.columns:
        raise ValueError("CSV must contain 'n' and 'd' columns for N/D annotations.")
    ns = df["n"].dropna().unique()
    ds = df["d"].dropna().unique()
    if len(ns) > 1 or len(ds) > 1:
        print(
            f"Warning: multiple n or d values in CSV; annotating N={int(ns[0])}, D={int(ds[0])}",
            file=sys.stderr,
        )
    return int(ns[0]), int(ds[0])


def add_n_d_annotation(ax, n_points: int, dim: int) -> None:
    """N and D in axes coordinates, same style as plot_outliers_benchmark.py."""
    ax.text(
        0.02,
        0.36,
        f"N={n_points}",
        transform=ax.transAxes,
        fontsize=30,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax.text(
        0.02,
        0.24,
        f"D={dim}",
        transform=ax.transAxes,
        fontsize=30,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_sensitivity_subplot(
    ax,
    agg: pd.DataFrame,
    score_col: str,
    distances: list[str],
    colors,
    markers,
    linestyles,
    show_ylabel: bool,
    title: str,
) -> list[Line2D]:
    legend_handles: list[Line2D] = []
    t_ticks = sorted(agg["time_tick"].unique())
    if len(t_ticks) > 0:
        ax.set_xticks(t_ticks)
        ax.xaxis.set_major_formatter(make_blend_time_tick_formatter(t_ticks))

    for i, dist in enumerate(distances):
        sub = agg[agg["distance_function"] == dist].sort_values("time_tick")
        if sub.empty or score_col not in sub.columns:
            continue
        x = sub["time_tick"].to_numpy()
        y = sub[score_col].to_numpy()
        color = colors[i % len(colors)]
        ls = linestyles[i % len(linestyles)]
        mk = markers[i % len(markers)]

        ax.plot(
            x,
            y,
            linestyle=ls,
            linewidth=5.6,
            color=color,
            alpha=0.9,
            zorder=2,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=ls,
                linewidth=2.5,
                marker=mk,
                markersize=7,
                markeredgewidth=1.5,
                markeredgecolor="white",
                label=DISTANCE_LABELS.get(dist, dist),
            )
        )

    for i, dist in enumerate(distances):
        sub = agg[agg["distance_function"] == dist].sort_values("time_tick")
        if sub.empty or score_col not in sub.columns:
            continue
        x = sub["time_tick"].to_numpy()
        y = sub[score_col].to_numpy()
        color = colors[i % len(colors)]
        mk = markers[i % len(markers)]
        marker_offset = i % 3
        ax.plot(
            x,
            y,
            marker=mk,
            linestyle="",
            markersize=28,
            color=color,
            markeredgewidth=2.8,
            markeredgecolor="white",
            markevery=(marker_offset, 3),
            zorder=10,
        )

    ax.set_xlabel(r"Time $t$", fontsize=30, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("Similarity score", fontsize=30, fontweight="bold")
    ax.set_title(title, fontsize=30, fontweight="bold")
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=4)
    return legend_handles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot TSI/QSI sensitivity to distance selection from experiment CSVs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing tsi_qsi_sensitivity_to_distance_selection*.csv. "
            "Default: experiments/results/tsi_qsi_sensitivity_to_distance_selection "
            "(same as the experiment script output). "
            "Use a path like .../data/tsi_qsi_sensitivity_to_distance_selection if you store copies there."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Use a specific CSV file instead of the first match in results-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default="../../plots",
        help="Output directory for figures (default: ../../plots under plot-rendering-code).",
    )
    parser.add_argument(
        "--output-name",
        default="tsi_qsi_sensitivity_to_distance_selection.png",
        help="Output filename (default: tsi_qsi_sensitivity_to_distance_selection.png).",
    )
    args = parser.parse_args()

    if args.results_dir is not None:
        results_dir = args.results_dir.expanduser().resolve()
    else:
        results_dir = (Path(__file__).parent.parent / "results" / DEFAULT_RESULTS_SUBDIR).resolve()

    if args.csv is not None:
        csv_path = args.csv.expanduser().resolve()
        if not csv_path.is_file():
            print(f"CSV not found: {csv_path}", file=sys.stderr)
            sys.exit(1)
    else:
        if not results_dir.is_dir():
            print(f"Results directory not found: {results_dir}", file=sys.stderr)
            print("Run experiments/scripts/tsi_qsi_sensitivity_to_distance_selection.py first.", file=sys.stderr)
            sys.exit(1)
        csv_files = find_csv_files(results_dir)
        if not csv_files:
            print(f"No CSV files in {results_dir}", file=sys.stderr)
            sys.exit(1)
        csv_path = csv_files[0]
        print(f"Using CSV: {csv_path.name}")

    df = pd.read_csv(csv_path)
    required = {"distance_function", "time_tick"}
    if not required.issubset(df.columns):
        print(f"CSV missing required columns {required}. Found: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)
    has_tsi = "tsi_score" in df.columns
    has_qsi = "qsi_score" in df.columns
    if not has_tsi and not has_qsi:
        print("CSV must contain tsi_score and/or qsi_score.", file=sys.stderr)
        sys.exit(1)

    agg = aggregate_by_distance_tick(df)
    n_points, dim = infer_n_d(df)
    distances = ordered_distances(df["distance_function"].unique().tolist())

    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"]

    n_subplots = int(has_tsi) + int(has_qsi)
    if n_subplots == 0:
        sys.exit(1)
    if n_subplots == 1:
        fig, ax_single = plt.subplots(1, 1, figsize=(16, 7))
        if has_tsi:
            legend_handles = plot_sensitivity_subplot(
                ax_single,
                agg,
                "tsi_score",
                distances,
                colors,
                markers,
                linestyles,
                show_ylabel=True,
                title="TSI",
            )
        else:
            legend_handles = plot_sensitivity_subplot(
                ax_single,
                agg,
                "qsi_score",
                distances,
                colors,
                markers,
                linestyles,
                show_ylabel=True,
                title="QSI",
            )
        ax_single.set_ylim(0.48, 1.02)
        apply_similarity_score_y_axis(ax_single)
        add_n_d_annotation(ax_single, n_points, dim)
        leg = ax_single.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0.0,
            fontsize=28,
            framealpha=0.95,
            edgecolor="black",
            fancybox=False,
            shadow=False,
            ncol=1,
        )
        for lh in leg.legend_handles:
            lh.set_linewidth(6.4)
            lh.set_markersize(28)
        plt.tight_layout(rect=[0, 0, 0.88, 1])
    else:
        fig, (ax_tsi, ax_qsi) = plt.subplots(1, 2, figsize=(32, 7), sharey=True)
        legend_handles = plot_sensitivity_subplot(
            ax_tsi,
            agg,
            "tsi_score",
            distances,
            colors,
            markers,
            linestyles,
            show_ylabel=True,
            title="TSI",
        )
        plot_sensitivity_subplot(
            ax_qsi,
            agg,
            "qsi_score",
            distances,
            colors,
            markers,
            linestyles,
            show_ylabel=False,
            title="QSI",
        )
        ax_tsi.set_ylim(0.48, 1.02)
        apply_similarity_score_y_axis(ax_tsi)
        add_n_d_annotation(ax_tsi, n_points, dim)
        add_n_d_annotation(ax_qsi, n_points, dim)
        leg = ax_qsi.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0.0,
            fontsize=28,
            framealpha=0.95,
            edgecolor="black",
            fancybox=False,
            shadow=False,
            ncol=1,
        )
        for lh in leg.legend_handles:
            lh.set_linewidth(6.4)
            lh.set_markersize(28)
        plt.tight_layout(rect=[0, 0, 0.95, 1])

    if args.output_dir.startswith("../"):
        output_dir = Path(__file__).parent / args.output_dir
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    out_path = output_dir / args.output_name
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
