import time
import numpy as np
import similarity as sm

BASELINE_MEASURES = {
    "CKA": "measure/representation_similarity/cka-kernel=linear-hsic=gretton-score",
	"CKNNA": "measure/platonic/cknna-topk={topk}",
    "SVCCA": "measure/svcca/cca-score",
    "PWCCA": "measure/svcca/pwcca-score",
    "Kendall-RDM": "measure/rsatoolbox/rsa-rdm=squared_euclidean-compare=tau_a",
    "OrthogonalProcrustes": "measure/sim_metric/procrustes-score",
}

def normalize_measure_name(name: str, value: float):
    if name == "Kendall-RDM":
        return (value + 1) / 2
    else:
        return value

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
    for measure_name, measure_path in BASELINE_MEASURES.items():
        if time_monitor:
            score, time_taken = BaselineMeasure(measure_path)(X, Y, time_monitor)
            results[measure_name] = (normalize_measure_name(measure_name, score), time_taken)
        else:
            results[measure_name] = normalize_measure_name(measure_name, BaselineMeasure(measure_path)(X, Y))
    return results