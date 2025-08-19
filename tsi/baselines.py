import time
import numpy as np
import similarity as sm

BASELINE_MEASURES = [
    "platonic/cka",
	"platonic/cka_rbf",
	"platonic/cknna_topk",
	"platonic/cycle_knn_topk",
	"platonic/edit_distance_knn_topk",
	"platonic/lcs_knn_topk",
	"platonic/mutual_knn_topk",
	"platonic/svcca",
	"platonic/unbiased_cka",
	"platonic/unbiased_cka_rbf"
]

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
    

def run_baseline_measures(X: np.ndarray, Y: np.ndarray, time_monitor: bool = False):
    results = {}
    for measure in BASELINE_MEASURES:
        results[measure] = BaselineMeasure(measure)(X, Y, time_monitor)
    return results