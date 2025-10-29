import numpy as np
import pymp
from src.data import RepresentationPair
from sklearn.metrics import pairwise_distances

from src.tsi import EfficientTSI
from src.utils import efficient_concordant_rank_computation

def qsi_predicate(i,j,k,l,X,Y,d_x,d_y):
    return np.sign(d_y(Y[i], Y[j]) - d_y(Y[k], Y[l])) == np.sign(d_x(X[i], X[j]) - d_x(X[k], X[l]))


class QSI:
    def __init__(self):
        self.name = "QSI"

    def __call__(self, representations: RepresentationPair):
        X, Y, d_x, d_y = representations.X, representations.Y, representations.d_x, representations.d_y
        n = len(X)
        aligned_quadruplets = 0
        for i in range(n):
            for j in range(n):
                if j == i:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    for l in range(n):
                        if l == i or l == j or l == k:
                            continue
                        aligned_quadruplets += qsi_predicate(i, j, k, l, X, Y, d_x, d_y)
        return aligned_quadruplets / (n * (n - 1) * (n - 2) * (n - 3))
    

class EfficientQSI:
    """
    The EfficientQSI class is used to efficiently compute the QSI between two representations. This implementation depends on the distance function being symmetric.
    """
    def __init__(self, euclidean: bool = False):
        self.name = "EfficientQSI"
        self.euclidean = euclidean
        self.tsi_helper = EfficientTSI(euclidean=euclidean, memory_efficient=False)

    def __call__(self, representations: RepresentationPair):
        """
        Compute the efficient QSI between two representations. This implementation depends on the distance function being symmetric.
        """
        X, Y, d_x, d_y = representations.X, representations.Y, representations.d_x, representations.d_y
        n = len(X)

        tsi_score = self.tsi_helper(representations)
        
        metric_x = 'euclidean' if self.euclidean else d_x
        metric_y = 'euclidean' if self.euclidean else d_y

        k_x = pairwise_distances(X, metric=metric_x, n_jobs=-1)
        k_y = pairwise_distances(Y, metric=metric_y, n_jobs=-1)
        
        upper_triangle_mask = np.triu(np.ones((n, n), dtype=bool), k=1).astype(bool)

        full_distances_x = k_x[upper_triangle_mask].flatten()
        full_distances_y = k_y[upper_triangle_mask].flatten()

        no_elements = (n * (n - 1)) / 2

        adjusted_kendall_tau = efficient_concordant_rank_computation(distances_x=full_distances_x, distances_y=full_distances_y) / (no_elements * (no_elements - 1))
        
        return (adjusted_kendall_tau - (4 / (n+1)) * tsi_score)  * ((n+1) / (n-3))

class ApproxQSI:
    """
    The ApproxQSI class is used to compute the approximate QSI between two representations 
    with a given error probability delta and a maximum additive error epsilon.
    """
    def __init__(self, n_samples: int | None = None, epsilon: float | None = None, delta: float | None = None, n_threads: int = 8, seed: int = 42):
        if n_samples is None and (epsilon is None and delta is None):
            raise ValueError("Either n_samples or (epsilon, delta) must be provided")
        self.name = "ApproxQSI"
        self.n_samples = n_samples
        self.epsilon = epsilon
        self.delta = delta
        self.n_threads = n_threads
        self.seed = seed

    def __call__(self, representations: RepresentationPair):
        X, Y, d_x, d_y = representations.X, representations.Y, representations.d_x, representations.d_y
        n = len(X)
        if self.n_samples is not None:
            comparisons = self.n_samples
        else:
            epsilon_term = 1/(2*(self.epsilon**2))
            delta_term = np.log(2/self.delta)
            comparisons = int(np.ceil(epsilon_term * delta_term))
        
        np.random.seed(self.seed)
        samples = []
        while len(samples) < comparisons:
            i, j, k, l = np.random.randint(0, n, 4)
            if i != j and i != k and i != l and j != k and j != l and k != l:
                samples.append((i, j, k, l))
        
        # Parallel computation
        aligned_quadruplets = pymp.shared.array((comparisons,), dtype=np.int32)
        with pymp.Parallel(self.n_threads) as p:
            for idx in p.range(comparisons):
                i, j, k, l = samples[idx]
                aligned_quadruplets[idx] = qsi_predicate(i, j, k, l, X, Y, d_x, d_y)
        
        return aligned_quadruplets.sum() / comparisons


class EfficientApproxQSI:
    """
    The EfficientApproxQSI class is used to efficiently compute the approximate QSI between two representations but without mathematical guarantees.
    """
    def __init__(self, euclidean: bool = False, n_threads: int = 8, batch_size: int = 100, no_batches: int = 10, seed: int = 42):
        self.name = "EfficientApproxQSI"
        self.n_threads = n_threads
        self.batch_size = batch_size
        self.no_batches = no_batches
        self.seed = seed
        self.qsi_helper = EfficientQSI(euclidean=euclidean)

    def __call__(self, representations: RepresentationPair):
        X, Y, d_x, d_y = representations.X, representations.Y, representations.d_x, representations.d_y
        n = len(X)
        if self.batch_size >= n:
            return self.qsi_helper(representations)
        
        np.random.seed(self.seed)
        batches = [np.random.choice(n, self.batch_size, replace=False) for _ in range(self.no_batches)]
        results = []
        
        for batch in batches:
            batch_representations = RepresentationPair(X=X[batch], Y=Y[batch], d_x=d_x, d_y=d_y)
            results.append(self.qsi_helper(batch_representations))
        
        return np.mean(results)
    