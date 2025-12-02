import time
import numpy as np
import similarity as sm
from src.cka_helper import gram_linear, center_gram

BASELINE_MEASURES = {
    "CKA": "measure/representation_similarity/cka-kernel=linear-hsic=gretton-score",
	"CKNNA": "measure/platonic/cknna-topk={topk}",
    "MutualNN": "measure/platonic/mutual_knn-topk={topk}",
    "SVCCA": "measure/svcca/cca-score",
    "PWCCA": "measure/svcca/pwcca-score",
}

class BaselineMeasure:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, X: np.ndarray, Y: np.ndarray, time_monitor: bool = False):
        if time_monitor:
            start_time = time.time()
            score = sm.make(self.name)(X, Y)
            end_time = time.time()
            return score, end_time - start_time
        else:
            return sm.make(self.name)(X, Y)

class ApproximateCKA:
    """ Inspired from the paper https://arxiv.org/abs/2010.15327"""
    def __init__(self):
        self.name = "AP-S-CKA"
    
    def unbiased_hsic(self, gram_x, gram_y):
        def _multiply_by_ones(matrix):
            return np.ones((1, n)) @ matrix @ np.ones((n, 1))
        n = gram_x.shape[0]
        gram_x_tilde = gram_x.copy()
        gram_y_tilde = gram_y.copy()
        np.fill_diagonal(gram_x_tilde, 0)
        np.fill_diagonal(gram_y_tilde, 0)
        first_term = np.trace(gram_x_tilde @ gram_y_tilde)
        second_term = (1 / ((n-1)*(n-2))) * (_multiply_by_ones(gram_x_tilde) @ _multiply_by_ones(gram_y_tilde))[0,0]
        third_term = -(2/(n-2)) * (_multiply_by_ones(gram_x_tilde @ gram_y_tilde))[0,0]
        return (1/(n*(n-3))) * (first_term + second_term + third_term)

    def __call__(self, X: np.ndarray, Y: np.ndarray, batch_size: int = 100, no_batches: int = 10, seed: int = 42):
        n = len(X)
        
        np.random.seed(seed)
        batches = [np.random.choice(n, batch_size, replace=False) for _ in range(no_batches)]

        cross_hsic_sum = 0
        hsic_x_sum = 0
        hsic_y_sum = 0

        for batch in batches:
            gram_linear_x = gram_linear(X[batch])
            gram_linear_y = gram_linear(Y[batch])
            
            cross_hsic_sum += self.unbiased_hsic(gram_linear_x, gram_linear_y)/no_batches
            hsic_x_sum += self.unbiased_hsic(gram_linear_x, gram_linear_x)/no_batches
            hsic_y_sum += self.unbiased_hsic(gram_linear_y, gram_linear_y)/no_batches
        
        return cross_hsic_sum / (np.sqrt(hsic_x_sum) * np.sqrt(hsic_y_sum))


class ApproximateBaselineMeasure:
    def __init__(self, name: str):
        self.name = f"AP-{name}"
        self.baseline_measure = BaselineMeasure(name)

    def __call__(self, X: np.ndarray, Y: np.ndarray, batch_size: int = 100, no_batches: int = 10, seed: int = 42):
        n = len(X)
        if batch_size >= n:
            return self.baseline_measure(X, Y)
        
        np.random.seed(seed)
        batches = [np.random.choice(n, batch_size, replace=False) for _ in range(no_batches)]
        results = []
        
        for batch in batches:
            batch_X = X[batch]
            batch_Y = Y[batch]
            results.append(self.baseline_measure(batch_X, batch_Y))
        
        return np.mean(results)

def run_baseline_measures(X: np.ndarray, Y: np.ndarray, time_monitor: bool = False):
    results = {}
    for measure_name, measure_path in BASELINE_MEASURES.items():
        try:
            if time_monitor:
                score, time_taken = BaselineMeasure(measure_path)(X, Y, time_monitor)
                results[measure_name] = (score, time_taken)
            else:
                results[measure_name] = BaselineMeasure(measure_path)(X, Y)
        except Exception as e:
            if time_monitor:
                results[measure_name] = (None, None)
            else:
                results[measure_name] = None
            print(f"Error running {measure_name} due to Error: {e}")
    return results

def run_approximate_baseline_measures(X: np.ndarray, Y: np.ndarray, batch_size: int = 100, no_batches: int = 10, seed: int = 42):
    results = {}
    for measure_name, measure_path in BASELINE_MEASURES.items():
        try:
            metric = ApproximateBaselineMeasure(measure_path)
            results[f"B-{measure_name}"] = metric(X, Y, batch_size, no_batches, seed)
        except Exception as e:
            results[f"B-{measure_name}"] = None
            print(f"Error running {measure_name} due to Error: {e}")
    results["C-CKA"] = ApproximateCKA()(X, Y, batch_size, no_batches, seed)
    return results