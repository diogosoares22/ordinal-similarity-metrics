#!/usr/bin/env python3
"""
Compute TSI/QSI + baselines (and optional permutation null)
from saved REAL probe features.

Input structure (produced by your training script):
 out_dir/features/seed_{A}/epoch_*/block{b}_{cls|mean}.npy
 out_dir/features/seed_{B}/epoch_*/block{b}_{cls|mean}.npy

Output:
 out_dir/similarity_scores.csv (or custom via --out-csv)

Typical usage for ONLY random init (epoch 0):
python analyze_rep_convergence_pair.py \
  --out-dir /p/scratch/.../vit_c10_convergence \
  --seed-a 0 --seed-b 1 \
  --epochs 0 \
  --blocks 3 6 9 12 \
  --poolings cls mean \
  --tsi-mode approx --tsi-n-samples 200000 \
  --qsi-mode approx --qsi-n-samples 200000 \
  --baseline-subsample 3000 \
  --n-perm 0 \
  --wandb-project ordinal-similarity \
  --wandb-run-name vit-c10-epoch0
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from src.baselines import run_baseline_measures
from src.data import RepresentationPair

# --- TSI variants (your code) ---
from src.tsi import EfficientTSI, ApproxTSI, EfficientApproxTSI

# --- QSI variants (your code) ---
from src.qsi import EfficientQSI, ApproxQSI, EfficientApproxQSI


# --------------------------- helpers ---------------------------

def _make_reps(X: np.ndarray, Y: np.ndarray) -> RepresentationPair:
    d = lambda a, b: np.linalg.norm(a - b)
    return RepresentationPair(X=X, Y=Y, d_x=d, d_y=d)


def _maybe_subsample(X: np.ndarray, Y: np.ndarray, n: Optional[int], seed: int = 0):
    """
    Optional hard subsampling to guarantee speed regardless of metric implementation.
    If n is None or n >= len(X), returns original.
    """
    if n is None or n >= len(X):
        return X, Y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), n, replace=False)
    return X[idx], Y[idx]


# --------------------------- metric factories ---------------------------

def make_tsi(args):
    """
    Build a TSI callable based on --tsi-mode.
    Modes:
      - approx            -> ApproxTSI (MC triplets with guarantees if eps/delta given)
      - efficient_approx  -> EfficientApproxTSI (batch averaging)
      - efficient         -> EfficientTSI
    """
    mode = args.tsi_mode

    if mode == "approx":
        return ApproxTSI(
            n_samples=args.tsi_n_samples,
            epsilon=args.tsi_epsilon,
            delta=args.tsi_delta,
            n_threads=args.tsi_threads,
            seed=args.tsi_seed,
        )

    if mode == "efficient_approx":
        return EfficientApproxTSI(
            euclidean=True,
            memory_efficient=True,
            n_threads=args.tsi_threads,
            batch_size=args.tsi_batch_size,
            no_batches=args.tsi_no_batches,
            seed=args.tsi_seed,
        )

    if mode == "efficient":
        return EfficientTSI(euclidean=True, memory_efficient=True)

    raise ValueError(f"Unknown --tsi-mode {mode}")


def make_qsi(args):
    """
    Build a QSI callable based on --qsi-mode.
    Modes:
      - approx            -> ApproxQSI (MC quadruplets with guarantees if eps/delta given)
      - efficient_approx  -> EfficientApproxQSI (batch averaging)
      - efficient         -> EfficientQSI
    """
    mode = args.qsi_mode

    if mode == "approx":
        return ApproxQSI(
            n_samples=args.qsi_n_samples,
            epsilon=args.qsi_epsilon,
            delta=args.qsi_delta,
            n_threads=args.qsi_threads,
            seed=args.qsi_seed,
        )

    if mode == "efficient_approx":
        return EfficientApproxQSI(
            euclidean=True,
            n_threads=args.qsi_threads,
            batch_size=args.qsi_batch_size,
            no_batches=args.qsi_no_batches,
            seed=args.qsi_seed,
        )

    if mode == "efficient":
        return EfficientQSI(euclidean=True)

    raise ValueError(f"Unknown --qsi-mode {mode}")


# --------------------------- compute metrics ---------------------------

def compute_metrics(
    X: np.ndarray,
    Y: np.ndarray,
    tsi_fn,
    qsi_fn,
    baseline_subsample: Optional[int] = None,
    baseline_seed: int = 0,
) -> Dict[str, float]:
    """
    Compute TSI/QSI (configured) + baselines.
    Optionally subsample ONLY for baselines to keep them cheap.
    """
    reps = _make_reps(X, Y)

    tsi = tsi_fn(reps)
    qsi = qsi_fn(reps)

    if baseline_subsample is not None:
        Xb, Yb = _maybe_subsample(X, Y, baseline_subsample, seed=baseline_seed)
        baseline_scores = run_baseline_measures(Xb, Yb, time_monitor=False)
    else:
        baseline_scores = run_baseline_measures(X, Y, time_monitor=False)

    out = {"TSI": float(tsi), "QSI": float(qsi)}
    out.update({k: float(v) if v is not None else np.nan for k, v in baseline_scores.items()})
    return out


def compute_perm_null_band(
    X: np.ndarray,
    Y: np.ndarray,
    n_perm: int,
    seed: int,
    tsi_fn,
    qsi_fn,
    baseline_subsample: Optional[int] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Permutation null: shuffle Y across samples, compute metrics n_perm times.
    Returns mean/std per metric.
    """
    rng = np.random.default_rng(seed)
    vals: Dict[str, List[float]] = {}

    for p in range(n_perm):
        perm = rng.permutation(len(Y))
        Yp = Y[perm]
        scores = compute_metrics(
            X, Yp,
            tsi_fn=tsi_fn,
            qsi_fn=qsi_fn,
            baseline_subsample=baseline_subsample,
            baseline_seed=seed + p,
        )
        for k, v in scores.items():
            vals.setdefault(k, []).append(v)

    return {
        k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
        for k, v in vals.items()
    }


# --------------------------- wandb ---------------------------

def wandb_init(project: Optional[str], run_name: Optional[str], config: dict):
    if project is None:
        return None
    try:
        import wandb
        wandb.init(project=project, name=run_name, config=config)
        return wandb
    except Exception as e:
        print(f"[W&B] disabled (could not init): {e}")
        return None


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--seed-a", type=int, default=0)
    ap.add_argument("--seed-b", type=int, default=1)
    ap.add_argument(
        "--epochs", type=int, required=True,
        help="Compute metrics for epochs 0..epochs. Use --epochs 0 for random init only."
    )
    ap.add_argument("--blocks", type=int, nargs="+", default=[3, 6, 9, 12])
    ap.add_argument("--poolings", type=str, nargs="+", default=["cls", "mean"],
                    choices=["cls", "mean"])

    # ---------- TSI config ----------
    ap.add_argument("--tsi-mode", type=str, default="approx",
                    choices=["approx", "efficient_approx", "efficient"])
    ap.add_argument("--tsi-n-samples", type=int, default=None,
                    help="ApproxTSI: number of triplets to sample.")
    ap.add_argument("--tsi-epsilon", type=float, default=None)
    ap.add_argument("--tsi-delta", type=float, default=None)
    ap.add_argument("--tsi-batch-size", type=int, default=1000,
                    help="EfficientApproxTSI: batch size.")
    ap.add_argument("--tsi-no-batches", type=int, default=10,
                    help="EfficientApproxTSI: number of batches.")
    ap.add_argument("--tsi-threads", type=int, default=8)
    ap.add_argument("--tsi-seed", type=int, default=42)

    # ---------- QSI config ----------
    ap.add_argument("--qsi-mode", type=str, default="approx",
                    choices=["approx", "efficient_approx", "efficient"])
    ap.add_argument("--qsi-n-samples", type=int, default=None,
                    help="ApproxQSI: number of quadruplets to sample.")
    ap.add_argument("--qsi-epsilon", type=float, default=None)
    ap.add_argument("--qsi-delta", type=float, default=None)
    ap.add_argument("--qsi-batch-size", type=int, default=1000,
                    help="EfficientApproxQSI: batch size.")
    ap.add_argument("--qsi-no-batches", type=int, default=10,
                    help="EfficientApproxQSI: number of batches.")
    ap.add_argument("--qsi-threads", type=int, default=8)
    ap.add_argument("--qsi-seed", type=int, default=42)

    # Baseline subsample
    ap.add_argument("--baseline-subsample", type=int, default=None,
                    help="Compute baselines on a random subset of this size (speed).")

    # Perm null / output
    ap.add_argument("--n-perm", type=int, default=0,
                    help="Permutation null repeats. Set 0 to skip null.")
    ap.add_argument("--out-csv", type=str, default=None)

    # W&B
    ap.add_argument("--wandb-project", type=str, default=None)
    ap.add_argument("--wandb-run-name", type=str, default=None)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    feat_a = out_dir / "features" / f"seed_{args.seed_a}"
    feat_b = out_dir / "features" / f"seed_{args.seed_b}"

    wb = wandb_init(args.wandb_project, args.wandb_run_name, config=vars(args))

    tsi_fn = make_tsi(args)
    qsi_fn = make_qsi(args)

    print(f"[INFO] Using TSI = {tsi_fn.__class__.__name__} | mode={args.tsi_mode}")
    print(f"[INFO] Using QSI = {qsi_fn.__class__.__name__} | mode={args.qsi_mode}")

    rows = []
    for epoch in range(0, args.epochs + 1):
        for b in args.blocks:
            for pooling in args.poolings:
                fp_a = feat_a / f"epoch_{epoch}" / f"block{b}_{pooling}.npy"
                fp_b = feat_b / f"epoch_{epoch}" / f"block{b}_{pooling}.npy"

                if not fp_a.exists() or not fp_b.exists():
                    raise FileNotFoundError(f"Missing feature file(s): {fp_a} or {fp_b}")

                Xa = np.load(fp_a).astype(np.float32, copy=False)
                Yb = np.load(fp_b).astype(np.float32, copy=False)

                scores = compute_metrics(
                    Xa, Yb,
                    tsi_fn=tsi_fn,
                    qsi_fn=qsi_fn,
                    baseline_subsample=args.baseline_subsample,
                    baseline_seed=epoch * 1000 + b,
                )

                if args.n_perm > 0:
                    null_band = compute_perm_null_band(
                        Xa, Yb,
                        n_perm=args.n_perm,
                        seed=epoch * 1000 + b,
                        tsi_fn=tsi_fn,
                        qsi_fn=qsi_fn,
                        baseline_subsample=args.baseline_subsample,
                    )
                else:
                    null_band = {k: (np.nan, np.nan) for k in scores.keys()}

                for k, v in scores.items():
                    nm, ns = null_band.get(k, (np.nan, np.nan))
                    rows.append({
                        "epoch": epoch,
                        "block": b,
                        "pooling": pooling,
                        "metric": k,
                        "value": float(v),
                        "null_mean": float(nm) if nm == nm else np.nan,
                        "null_std": float(ns) if ns == ns else np.nan,
                        "seed_a": args.seed_a,
                        "seed_b": args.seed_b,
                        "tsi_mode": args.tsi_mode,
                        "qsi_mode": args.qsi_mode,
                        "tsi_n_samples": args.tsi_n_samples,
                        "qsi_n_samples": args.qsi_n_samples,
                        "tsi_batch_size": args.tsi_batch_size,
                        "qsi_batch_size": args.qsi_batch_size,
                        "tsi_no_batches": args.tsi_no_batches,
                        "qsi_no_batches": args.qsi_no_batches,
                        "n_perm": args.n_perm,
                        "baseline_subsample": args.baseline_subsample,
                    })

        if wb is not None:
            ep_df = pd.DataFrame([r for r in rows if r["epoch"] == epoch])
            for metric in ep_df["metric"].unique():
                mvals = ep_df[ep_df["metric"] == metric]["value"].values
                wb.log({f"{metric}/mean_over_blocks_poolings": float(np.mean(mvals))}, step=epoch)
            print(f"[Epoch {epoch}] logged to W&B.")

    df = pd.DataFrame(rows)

    out_csv = Path(args.out_csv) if args.out_csv else (out_dir / "similarity_scores_epochs.csv")
    df.to_csv(out_csv, index=False)
    print(f"[Done] wrote {out_csv}")
    print(df.head())

    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()