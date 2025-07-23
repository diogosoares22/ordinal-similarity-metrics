"""
This module defines the Triplet Similarity Index (TSI)

The TSI measures the similarity between two representations

\text{TSI}_{d_X, d_Y}(X, Y) = \mathbb{E}_{i,j,k \sim \mathcal{T}_{\{1, \dots, N\}}} \left[ \mathbb{I}\Big[ \sign\big(d_Y(y_i, y_j) - d_Y(y_i, y_k)\big) = \sign\big(d_X(x_i, x_j) - d_X(x_i, x_k)\big) \Big] \right]
"""

import numpy as np
from scipy.stats._stats_py import _kendall_dis


class TSI:
    """
    The TSI class is used to compute the TSI between two representations.
    """
    def __init__(self, n=None):
        pass

    def __call__(self, X, Y, d_x, d_y):
        """
        Compute the TSI between two representations. Let X and Y be two sets of points in a metric space.
        Let d_X and d_Y be two distance functions on X and Y respectively.
        Let T be a set of triplets of indices (i, j, k) where i, j, k are in {1, ..., N} and i != j != k.
        """
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
    

class ApproxTSI:
    """
    The ApproxTSI class is used to compute the approximate TSI between two representations.
    """
    def __init__(self, k=None):
        self.k = k

    def __call__(self, X, Y, d_x, d_y):
        """
        Compute the approximate TSI between two representations.
        """
        n = len(X)
        samples = [np.random.choice(n, size=3, replace=False) for _ in range(self.k)]
        aligned_triplets = 0
        for sample in samples:
            aligned_triplets += np.sign(d_y(Y[sample[0]], Y[sample[1]]) - d_y(Y[sample[0]], Y[sample[2]])) == np.sign(d_x(X[sample[0]], X[sample[1]]) - d_x(X[sample[0]], X[sample[2]]))
        return aligned_triplets / self.k
    

class _FenwickTree:
    """A Fenwick Tree for suffix updates and point queries."""
    def __init__(self, size):
        self.tree = [0] * (size + 1)

    def update(self, i, delta):
        i += 1
        while i < len(self.tree):
            self.tree[i] += delta
            i += i & (-i)

    def query(self, i):
        i += 1
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s
    

class EfficientTSI:
    """
    The EfficientTSI class is used to efficiently compute the TSI between two representations.
    """
    def __init__(self):
        pass

    def _compute_inversions_and_ties(self, distances_x, distances_y):
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

        dis = _kendall_dis(x, y)  # discordant pairs

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

    def __call__(self, X, Y, d_x, d_y):
        """
        Compute the efficient TSI between two representations.
        """
        n = len(X)
        aligned_triplets = 0
        for i in range(n):
            anchor_point = i
            distances_x = np.array([d_x(X[anchor_point], X[j]) for j in list(range(0, anchor_point)) + list(range(anchor_point + 1, n))])
            distances_y = np.array([d_y(Y[anchor_point], Y[j]) for j in list(range(0, anchor_point)) + list(range(anchor_point + 1, n))])
            pos, _ = self._compute_inversions_and_ties(distances_x, distances_y)
            aligned_triplets += 2 * pos
        return aligned_triplets / (n * (n - 1) * (n - 2))