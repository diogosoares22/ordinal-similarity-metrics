import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats._stats import _kendall_dis


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