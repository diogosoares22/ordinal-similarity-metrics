#!/usr/bin/env python3
"""
Script to compare representations across different seeds.

Computes similarity between the final representation of seed i 
and the intermediate representations of seed (i+1) mod n_seeds.

This helps analyze how similar the training trajectories are across different
random initializations.
"""

import numpy as np
import os
import argparse
import pandas as pd
import time

from src.baselines import run_approximate_baseline_measures
from src.tsi import ApproxTSI
from src.qsi import ApproxQSI
from src.data import RepresentationPair


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare representations across seeds')
    parser.add_argument('--net', default='vit', choices=['vit', 'res50'], help='network architecture')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'], help='dataset')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate used in training')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs used in training')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4], help='list of seeds to compare')
    parser.add_argument('--similarity-bs', default=1000, type=int, help='batch size for similarity computation')
    parser.add_argument('--no-similarity-bs', default=10, type=int, help='number of batches for similarity computation')
    parser.add_argument('--custom-initialization', action='store_true', help='if custom initialization was used')
    parser.add_argument('--init-gain', default=1, type=float, help='gain for weight initialization (if custom)')
    return parser.parse_args()


def get_experiment_dir(args, seed):
    """Get the experiment directory for a given seed."""
    if args.custom_initialization:
        experiment_name = f"{args.net}_{args.dataset}_custom_init_gain{args.init_gain}_lr{args.lr}_epochs{args.n_epochs}_seed{seed}"
    else:
        experiment_name = f"{args.net}_{args.dataset}_lr{args.lr}_epochs{args.n_epochs}_seed{seed}"
    return os.path.join("experiments", "results", "train_model_and_compare_representations", experiment_name)


def load_representations(repr_path):
    """Load representations from a saved .npz file."""
    print(f"  Loading: {repr_path}")
    data = np.load(repr_path)
    epoch_representations = {}
    for key in data.files:
        epoch_num = int(key.replace("epoch_", ""))
        epoch_representations[epoch_num] = data[key]
    return epoch_representations


def compute_similarity_scores(X, Y, batch_size, no_batches):
    """Compute similarity scores between two sets of representations."""
    results = run_approximate_baseline_measures(X, Y, batch_size=batch_size, no_batches=no_batches)
    approx_n_samples = (batch_size ** 2) * no_batches
    d = lambda a, b: np.linalg.norm(a - b)
    representations = RepresentationPair(X=X, Y=Y, d_x=d, d_y=d)
    approx_tsi_sampling = ApproxTSI(n_samples=approx_n_samples, n_threads=8)(representations)
    approx_qsi_sampling = ApproxQSI(n_samples=approx_n_samples, n_threads=8)(representations)
    results["C-TSI"] = approx_tsi_sampling
    results["C-QSI"] = approx_qsi_sampling
    return results


def main():
    args = parse_args()
    seeds = args.seeds
    n_seeds = len(seeds)
    
    print(f"==> Comparing representations across {n_seeds} seeds: {seeds}")
    print(f"    Net: {args.net}, Dataset: {args.dataset}, LR: {args.lr}, Epochs: {args.n_epochs}")
    
    # Load all representations
    print("\n==> Loading representations for all seeds...")
    all_representations = {}
    for seed in seeds:
        exp_dir = get_experiment_dir(args, seed)
        repr_path = os.path.join(exp_dir, "representations.npz")
        if not os.path.exists(repr_path):
            raise FileNotFoundError(f"Representations not found: {repr_path}")
        all_representations[seed] = load_representations(repr_path)
    
    # Get all epoch numbers (should be the same for all seeds)
    sample_seed = seeds[0]
    all_epochs = sorted(all_representations[sample_seed].keys())
    final_epoch = max(all_epochs)
    print(f"\n==> Found {len(all_epochs)} epochs, final epoch: {final_epoch}")
    
    # For each seed i, compare final representation with intermediate representations of seed (i+1) mod n_seeds
    all_results = []
    
    for i, seed_i in enumerate(seeds):
        seed_j = seeds[(i + 1) % n_seeds]
        print(f"\n==> Comparing: seed {seed_i} (final) vs seed {seed_j} (all epochs)")
        
        # Get final representation of seed i
        final_repr_i = all_representations[seed_i][final_epoch]
        
        # Compare with each epoch of seed j
        for epoch in all_epochs:
            start_time = time.time()
            repr_j = all_representations[seed_j][epoch]
            
            similarity_scores = compute_similarity_scores(
                final_repr_i, repr_j,
                batch_size=args.similarity_bs, 
                no_batches=args.no_similarity_bs
            )
            
            result_entry = {
                "seed_final": seed_i,
                "seed_trajectory": seed_j,
                "epoch_trajectory": epoch,
                "epoch_final": final_epoch,
                "computation_time": time.time() - start_time,
            }
            for metric_name, metric_value in similarity_scores.items():
                result_entry[metric_name] = metric_value
            
            all_results.append(result_entry)
            
            if epoch % 20 == 0 or epoch == -1:
                print(f"  Epoch {epoch}: " + ", ".join([f"{k}: {v:.4f}" for k, v in similarity_scores.items() if v is not None]))
    
    # Save results
    results_df = pd.DataFrame(all_results)
    
    # Save to the first seed's directory with a descriptive name
    output_dir = get_experiment_dir(args, seeds[0])
    seeds_str = "_".join(map(str, seeds))
    output_filename = f"cross_seed_similarity_seeds{seeds_str}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    results_df.to_csv(output_path, index=False)
    print(f"\n==> Results saved to: {output_path}")
    
    # Also save a summary CSV with just the final epoch comparisons
    final_comparisons = results_df[results_df["epoch_trajectory"] == final_epoch]
    summary_path = os.path.join(output_dir, f"cross_seed_summary_seeds{seeds_str}.csv")
    final_comparisons.to_csv(summary_path, index=False)
    print(f"==> Summary (final vs final) saved to: {summary_path}")


if __name__ == "__main__":
    main()

