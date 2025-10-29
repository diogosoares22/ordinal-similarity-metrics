import numpy as np
from src.data import RepresentationPair
from sklearn.metrics import pairwise_distances
import pymp
from src.utils import efficient_concordant_rank_computation

def tsi_predicate(i, j, k, X, Y, d_x, d_y):
    return np.sign(d_y(Y[i], Y[j]) - d_y(Y[i], Y[k])) == np.sign(d_x(X[i], X[j]) - d_x(X[i], X[k]))

class TSI:
    """
    The TSI class is used to compute the TSI between two representations.
    """
    def __init__(self):
        self.name = "TSI"

    def __call__(self, representations: RepresentationPair):
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
        self.name = "EfficientTSI"
        self.euclidean = euclidean
        self.memory_efficient = memory_efficient

    def __call__(self, representations: RepresentationPair):
        """
        Compute the efficient TSI between two representations.
        """
        X, Y, d_x, d_y = representations.X, representations.Y, representations.d_x, representations.d_y
        n = len(X)

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
                mask = None
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
    The ApproxTSI class is used to compute the approximate TSI between two representations 
    with a given error probability delta and a maximum additive error epsilon.
    """
    def __init__(self, n_samples: int | None = None, epsilon: float | None = None, delta: float | None = None, n_threads: int = 8, seed: int = 42):
        if n_samples is None and (epsilon is None and delta is None):
            raise ValueError("Either n_samples or (epsilon, delta) must be provided")
        self.name = "ApproxTSI"
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
            i, j, k = np.random.randint(0, n, 3)
            if i != j and i != k and j != k:
                samples.append((i, j, k))
        
        # Parallel computation
        aligned_triplets = pymp.shared.array((comparisons,), dtype=np.int32)
        with pymp.Parallel(self.n_threads) as p:
            for idx in p.range(comparisons):
                i, j, k = samples[idx]
                aligned_triplets[idx] = tsi_predicate(i, j, k, X, Y, d_x, d_y)
        
        return aligned_triplets.sum() / comparisons


class EfficientApproxTSI:
    """
    The EfficientApproxTSI class is used to efficiently compute the approximate TSI between two representations but without mathematical guarantees.
    """
    def __init__(self, euclidean: bool = False, memory_efficient: bool = True, n_threads: int = 8, batch_size: int = 100, no_batches: int = 10, seed: int = 42):
        self.name = "EfficientApproxTSI"
        self.n_threads = n_threads
        self.batch_size = batch_size
        self.no_batches = no_batches
        self.seed = seed
        self.tsi_helper = EfficientTSI(euclidean=euclidean, memory_efficient=memory_efficient)

    def __call__(self, representations: RepresentationPair):
        X, Y, d_x, d_y = representations.X, representations.Y, representations.d_x, representations.d_y
        n = len(X)
        if self.batch_size >= n:
            return self.tsi_helper(representations)
        
        np.random.seed(self.seed)
        batches = [np.random.choice(n, self.batch_size, replace=False) for _ in range(self.no_batches)]
        results = []
        
        for batch in batches:
            batch_representations = RepresentationPair(X=X[batch], Y=Y[batch], d_x=d_x, d_y=d_y)
            results.append(self.tsi_helper(batch_representations))
        
        return np.mean(results)