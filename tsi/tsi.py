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

def tsi_predicate_with_sign_evaluation(i, j, k, X=None, Y=None, d_x=None, d_y=None, sign_x_evaluation=None, sign_y_evaluation=None):
    if sign_x_evaluation is None:
        sign_x_evaluation = np.sign(d_x(X[i], X[j]) - d_x(X[i], X[k]))
    if sign_y_evaluation is None:
        sign_y_evaluation = np.sign(d_y(Y[i], Y[j]) - d_y(Y[i], Y[k]))
    return sign_x_evaluation == sign_y_evaluation

def tsi_predicate(i, j, k, X, Y, d_x, d_y):
    return np.sign(d_y(Y[i], Y[j]) - d_y(Y[i], Y[k])) == np.sign(d_x(X[i], X[j]) - d_x(X[i], X[k]))

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

def efficient_concordant_rank_computation(anchor=None, mask=None, X=None, Y=None, d_x=None, d_y=None, distances_x=None, distances_y=None):
    if (anchor is None or mask is None or X is None or Y is None or d_x is None or d_y is None) and (distances_x is None or distances_y is None):
        raise ValueError("Either anchor, mask, X, Y, d_x, d_y must be provided or distances_x and distances_y must be provided")
    if distances_x is None:
        distances_x = pairwise_distances(X[[anchor]], X[mask], metric=d_x)[0]
    if distances_y is None:
        distances_y = pairwise_distances(Y[[anchor]], Y[mask], metric=d_y)[0]
    pos, _ = _compute_inversions_and_ties(distances_x, distances_y)
    return 2 * pos

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
                    aligned_triplets += tsi_predicate(i, j, k, X, Y, d_x, d_y)
        return aligned_triplets / (n * (n - 1) * (n - 2))    
    

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
                distances_x = None
                distances_y = None
                if not self.memory_efficient:
                    distances_x = full_distances_x[i, :]
                    distances_y = full_distances_y[i, :]
                else:
                    mask = np.ones(n, dtype=bool)
                    mask[i] = False
                results[i] = efficient_concordant_rank_computation(anchor=i, mask=mask, X=X, Y=Y, d_x=metric_x, d_y=metric_y, distances_x=distances_x, distances_y=distances_y)
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
                aligned_triplets += tsi_predicate(triplet[0], triplet[1], triplet[2], X, Y, d_x, d_y)
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
                    results[anchor] = efficient_concordant_rank_computation(anchor=anchor, mask=mask, X=X, Y=Y, d_x=metric_x, d_y=metric_y)
                    total_triplets[anchor] = len(complete_indices) * (len(complete_indices) - 1)
            return results.sum() / total_triplets.sum()
        else:
            raise ValueError("Invalid indices type")
        
# TODO: optimize this for time efficiency
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
    

class OddOneOutTSI:
    """
    The OddOneOutTSI class is used to compute TSI between one representation and an implicit representation using odd-one-out observations.
    """
    def __init__(self, odd_one_out_observations: dict[tuple[int, int, int], int]):
        self.odd_one_out_observations = odd_one_out_observations
    
    def _initialize_indices(self):
        indices_to_sign_evaluation = {}
        for triplet in self.odd_one_out_observations:
            odd = self.odd_one_out_observations[triplet]
            similar = list(triplet)
            similar.remove(odd)
            indices_to_sign_evaluation[(similar[0], odd, similar[1])] = 1
            indices_to_sign_evaluation[(similar[0], similar[1], odd)] = -1
            indices_to_sign_evaluation[(similar[1], odd, similar[0])] = 1
            indices_to_sign_evaluation[(similar[1], similar[0], odd)] = -1
        self.partial_indices = PartialIndices(indices=list(indices_to_sign_evaluation.keys()))
        self.sign_evaluations = indices_to_sign_evaluation

    def __call__(self, X: np.ndarray, d_x: Callable, initialize_indices: bool = True):
        """
        Compute the odd one out TSI between two representations.
        """
        if initialize_indices:
            self._initialize_indices()
        n = len(X)
        aligned_triplets = 0
        for indices in self.partial_indices.indices:
            aligned_triplets += tsi_predicate_with_sign_evaluation(indices[0], indices[1], indices[2], X=X, d_x=d_x, sign_y_evaluation=self.sign_evaluations[indices])
        return aligned_triplets / len(self.partial_indices.indices)