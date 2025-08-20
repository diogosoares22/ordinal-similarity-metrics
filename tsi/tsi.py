"""
This module defines the Triplet Similarity Index (TSI)

The TSI measures the similarity between two representations

\text{TSI}_{d_X, d_Y}(X, Y) = \mathbb{E}_{i,j,k \sim \mathcal{T}_{\{1, \dots, N\}}} \left[ \mathbb{I}\Big[ \sign\big(d_Y(y_i, y_j) - d_Y(y_i, y_k)\big) = \sign\big(d_X(x_i, x_j) - d_X(x_i, x_k)\big) \Big] \right]
"""

import numpy as np
from typing import Callable
from dataclasses import dataclass
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import pymp
from scipy.stats._stats import _kendall_dis

@dataclass
class RepresentationPair:
    X: np.ndarray
    Y: np.ndarray
    d_x: Callable
    d_y: Callable

@dataclass
class PartialIndices:
    indices: list[tuple[int, int, int]]
    
@dataclass
class CompleteIndices:
    indices: dict[int, list[int]]


class TSI:
    """
    The TSI class is used to compute the TSI between two representations.
    """
    def __init__(self):
        pass

    def __call__(self, representations: RepresentationPair):
        """
        Compute the TSI between two representations. Let X and Y be two sets of points in a metric space.
        Let d_X and d_Y be two distance functions on X and Y respectively.
        Let T be a set of triplets of indices (i, j, k) where i, j, k are in {1, ..., N} and i != j != k.
        """
        X, Y, d_x, d_y = representations.X, representations.Y, representations.d_x, representations.d_y
        n = len(X)
        aligned_triplets = 0
        for i in range(n):
            for j in range(n):
                if j == i:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    aligned_triplets += np.sign(d_y(Y[i], Y[j]) - d_y(Y[i], Y[k])) == np.sign(d_x(X[i], X[j]) - d_x(X[i], X[k]))
        return aligned_triplets / (n * (n - 1) * (n - 2))
    
def _compute_inversions_and_ties(distances_x, distances_y):
        """
        Compute the inversions and ties in the distances. Code inspired by scipy.stats.kendalltau.
        """
        x, y = distances_x, distances_y
        def count_rank_tie(ranks):
            cnt = np.bincount(ranks).astype('int64', copy=False)
            cnt = cnt[cnt > 1]
            # Python ints to avoid overflow down the line
            return (int((cnt * (cnt - 1) // 2).sum()),
                    int((cnt * (cnt - 1.) * (cnt - 2)).sum()),
                    int((cnt * (cnt - 1.) * (2*cnt + 5)).sum()))
        
        size = x.size
        perm = np.argsort(y)  # sort on y and convert y to dense ranks
        x, y = x[perm], y[perm]
        y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)

        # stable sort on x and convert x to dense ranks
        perm = np.argsort(x, kind='mergesort')
        x, y = x[perm], y[perm]
        x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

        dis = _kendall_dis(x, y) # discordant pairs

        obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
        cnt = np.diff(np.nonzero(obs)[0]).astype('int64', copy=False)

        ntie = int((cnt * (cnt - 1) // 2).sum())  # joint ties
        xtie, x0, x1 = count_rank_tie(x)     # ties in x, stats
        ytie, y0, y1 = count_rank_tie(y)     # ties in y, stats

        tot = (size * (size - 1)) // 2

        con = tot - dis - xtie - ytie + ntie

        pos = con + ntie

        neg = dis + (xtie - ntie) + (ytie - ntie)

        return pos, neg    
    

class EfficientTSI:
    """
    The EfficientTSI class is used to efficiently compute the TSI between two representations.
    """
    def __init__(self, euclidean: bool = False, memory_efficient: bool = True):
        self.euclidean = euclidean
        self.memory_efficient = memory_efficient

    def __call__(self, representations: RepresentationPair):
        """
        Compute the efficient TSI between two representations.
        """
        X, Y, d_x, d_y = representations.X, representations.Y, representations.d_x, representations.d_y
        n = len(X)
        aligned_triplets = 0

        metric_x = 'euclidean' if self.euclidean else d_x
        metric_y = 'euclidean' if self.euclidean else d_y

        if not self.memory_efficient:
            k_x = pairwise_distances(X, metric=metric_x, n_jobs=-1)
            k_y = pairwise_distances(Y, metric=metric_y, n_jobs=-1)
            
            full_mask = (np.ones((n, n), dtype=bool) - np.eye(n)).astype(bool)

            full_distances_x = np.array([k_x[i, full_mask[i, :]] for i in range(n)])
            full_distances_y = np.array([k_y[i, full_mask[i, :]] for i in range(n)])

        results = pymp.shared.array((n))
        with pymp.Parallel(8) as p:
            for i in p.range(n):
                if not self.memory_efficient:
                    distances_x = full_distances_x[i, :]
                    distances_y = full_distances_y[i, :]
                else:
                    mask = np.ones(n, dtype=bool)
                    mask[i] = False
                    distances_x = pairwise_distances(X[[i]], X[mask], metric=metric_x)[0]
                    distances_y = pairwise_distances(Y[[i]], Y[mask], metric=metric_y)[0]
                pos, _ = _compute_inversions_and_ties(distances_x, distances_y)
                results[i] = 2 * pos
        
        aligned_triplets = results.sum()
        return aligned_triplets / (n * (n - 1) * (n - 2))
    
class ApproxTSI:
    """
    The ApproxTSI class is used to compute the approximate TSI between two representations.
    """
    def __init__(self, euclidean: bool = False):
        self.euclidean = euclidean

    def __call__(self, representations: RepresentationPair, indices: PartialIndices|CompleteIndices):
        """
        Compute the approximate TSI between two representations.
        """
        X, Y, d_x, d_y = representations.X, representations.Y, representations.d_x, representations.d_y
        n = len(X)
        if isinstance(indices, PartialIndices):
            aligned_triplets = 0
            for triplet in indices.indices:
                aligned_triplets += np.sign(d_y(Y[triplet[0]], Y[triplet[1]]) - d_y(Y[triplet[0]], Y[triplet[2]])) == np.sign(d_x(X[triplet[0]], X[triplet[1]]) - d_x(X[triplet[0]], X[triplet[2]]))
            return aligned_triplets / len(indices.indices)
        elif isinstance(indices, CompleteIndices):
            results = pymp.shared.array((n))
            total_triplets = pymp.shared.array((n))
            metric_x = 'euclidean' if self.euclidean else d_x
            metric_y = 'euclidean' if self.euclidean else d_y
            with pymp.Parallel(8) as p:
                for anchor in p.iterate(indices.indices):
                    complete_indices = indices.indices[anchor]
                    mask = np.zeros(n, dtype=bool)
                    mask[complete_indices] = True
                    distances_x = pairwise_distances(X[[anchor]], X[mask], metric=metric_x)[0]
                    distances_y = pairwise_distances(Y[[anchor]], Y[mask], metric=metric_y)[0]
                    pos, _ = _compute_inversions_and_ties(distances_x, distances_y)
                    results[anchor] = 2 * pos
                    total_triplets[anchor] = len(complete_indices) * (len(complete_indices) - 1)
            return results.sum() / total_triplets.sum()
        else:
            raise ValueError("Invalid indices type")
        

class NearestNeighborTSI(ApproxTSI):
    """
    The NearestNeighborTSI class is used to compute the nearest neighbor TSI between two representations.
    """
    def __init__(self, euclidean: bool = False, k: int = 10):
        super().__init__(euclidean)
        self.k = k

    def __call__(self, representations: RepresentationPair):
        """
        Compute the nearest neighbor TSI between two representations.

        Note: the training data should be ordered by source input order. Otherwise, stable results are not guaranteed.
        """
        X, Y, d_x, d_y = representations.X, representations.Y, representations.d_x, representations.d_y
        n = len(X)
        nn_x = NearestNeighbors(n_neighbors=self.k + 1, metric=d_x)
        nn_y = NearestNeighbors(n_neighbors=self.k + 1, metric=d_y)
        nn_x.fit(X)
        nn_y.fit(Y)
        aligned_triplets = 0
        considered_triplets = 0
        for i in range(n):
            x_neighbors = nn_x.kneighbors(X[[i]], return_distance=False)[0]
            y_neighbors = nn_y.kneighbors(Y[[i]], return_distance=False)[0]
            x_neighbors = np.delete(x_neighbors, np.where(x_neighbors == i))
            y_neighbors = np.delete(y_neighbors, np.where(y_neighbors == i))
            union_neighbors = np.unique(np.concatenate((x_neighbors, y_neighbors)))
            aligned_triplets += super().__call__(representations, CompleteIndices(indices={i: union_neighbors})) * (len(union_neighbors) * (len(union_neighbors) - 1))
            considered_triplets += len(union_neighbors) * (len(union_neighbors) - 1)
        return aligned_triplets / considered_triplets
    
class BatchTSI(EfficientTSI):
    """
    The BatchTSI class is used to compute the batch TSI between two representations.
    """
    def __init__(self, euclidean: bool = False, memory_efficient: bool = True, batch_size: int = 100):
        super().__init__(euclidean, memory_efficient)
        self.batch_size = batch_size

    def __call__(self, representations: RepresentationPair):
        """
        Compute the batch TSI between two representations.
        """
        X, Y, d_x, d_y = representations.X, representations.Y, representations.d_x, representations.d_y
        n = len(X)
        aligned_triplets = 0
        considered_triplets = 0
        for i in range(0, n, self.batch_size):
            batch_x = X[i:i+self.batch_size]
            batch_y = Y[i:i+self.batch_size]
            actual_batch_size = len(batch_x)
            if actual_batch_size >= 3:
                batched_representations = RepresentationPair(X=batch_x, Y=batch_y, d_x=d_x, d_y=d_y)
                aligned_triplets += super().__call__(batched_representations) * (actual_batch_size * (actual_batch_size - 1))
                considered_triplets += actual_batch_size * (actual_batch_size - 1)
        return aligned_triplets / considered_triplets