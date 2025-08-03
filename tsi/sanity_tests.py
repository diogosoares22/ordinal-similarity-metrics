import time
from tsi import TSI, ApproxTSI, EfficientTSI, RepresentationPair
import numpy as np

class Example:
    def __init__(self, representations: RepresentationPair, expected_tsi, name):
        self.name = name
        self.representations = representations
        self.expected_tsi = expected_tsi

d_x = lambda x, y: np.linalg.norm(x - y)
d_y = lambda x, y: np.linalg.norm(x - y)

curated_example_1 = Example(RepresentationPair(X=np.array([[0, 0], [1, 0], [0, 1]]), Y=np.array([[0, 0], [1, 0], [1, 1]]), d_x=d_x, d_y=d_y), 0.0, "curated_example_1")
curated_example_2 = Example(RepresentationPair(X=np.array([[0, 0], [1, 0], [0, 1], [1, 1]]), Y=np.array([[0, 0], [1, 0], [0, 1], [2, 2]]), d_x=d_x, d_y=d_y), 2/3, "curated_example_2")
curated_example_3 = Example(RepresentationPair(X=np.array([[0, 0], [2, 0], [3, 0]]), Y=np.array([[0, 0], [2, 0], [1, 0]]), d_x=d_x, d_y=d_y), 1/3, "curated_example_3")

def test_tsi_on_example(example):
    tsi = TSI()
    assert tsi(example.representations) == example.expected_tsi
    print(f"TSI on example {example.name} test passed")

def test_tsi_on_random_data():
    X = np.random.rand(30, 3)
    Y = X
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    tsi = TSI()
    assert tsi(representations) == 1.0
    print("TSI on random data test passed")

def test_approx_tsi_alignment_with_tsi():
    X = np.random.rand(30, 3)
    Y = np.random.rand(30, 3)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    approx_tsi = ApproxTSI(k=20000)
    tsi = TSI()
    assert np.abs(approx_tsi(representations) - tsi(representations)) < 1e-2
    print("ApproxTSI alignment with TSI test passed")

def test_efficient_tsi_on_example(example):
    efficient_tsi = EfficientTSI(euclidean=True)
    assert efficient_tsi(example.representations) == example.expected_tsi
    print(f"EfficientTSI on example {example.name} test passed")

def test_efficient_tsi_alignment_with_tsi_on_random_data():
    X = np.random.rand(30, 3)
    Y = np.random.rand(30, 3)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    efficient_tsi = EfficientTSI(euclidean=True)
    tsi = TSI()
    assert efficient_tsi(representations) == tsi(representations)
    print("EfficientTSI alignment with TSI on random data test passed")

def test_efficient_tsi_alignment_with_tsi_on_random_data_with_equalities():
    X = np.random.rand(30, 3)
    Y = np.concatenate((X[:15], X[:15]), axis=0)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    efficient_tsi = EfficientTSI(euclidean=True)
    tsi = TSI()
    print(efficient_tsi(representations), tsi(representations))
    assert efficient_tsi(representations) == tsi(representations)
    print("EfficientTSI alignment with TSI test on random data with equalities passed")

def compare_tsi_and_efficient_tsi():
    X = np.random.rand(50, 20)
    Y = np.random.rand(50, 20)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    efficient_tsi = EfficientTSI(euclidean=True)
    tsi = TSI()
    start_time = time.time()
    tsi(representations)
    end_time = time.time()
    print(f"Time taken by TSI: {end_time - start_time}")
    start_time = time.time()
    efficient_tsi(representations)
    end_time = time.time()
    print(f"Time taken by EfficientTSI: {end_time - start_time}")

if __name__ == "__main__":
    test_tsi_on_example(curated_example_1)
    test_tsi_on_example(curated_example_2)
    test_tsi_on_example(curated_example_3)
    test_tsi_on_random_data()
    test_approx_tsi_alignment_with_tsi()
    test_efficient_tsi_on_example(curated_example_1)
    test_efficient_tsi_on_example(curated_example_2)
    test_efficient_tsi_on_example(curated_example_3)
    test_efficient_tsi_alignment_with_tsi_on_random_data()
    test_efficient_tsi_alignment_with_tsi_on_random_data_with_equalities()
    compare_tsi_and_efficient_tsi()