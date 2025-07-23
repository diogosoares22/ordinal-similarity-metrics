import time
from tsi import TSI, ApproxTSI, EfficientTSI
import numpy as np

class Example:
    def __init__(self, X, Y, expected_tsi, name):
        self.name = name
        self.X = X
        self.Y = Y
        self.expected_tsi = expected_tsi

curated_example_1 = Example(np.array([[0, 0], [1, 0], [0, 1]]), np.array([[0, 0], [1, 0], [1, 1]]), 0.0, "curated_example_1")
curated_example_2 = Example(np.array([[0, 0], [1, 0], [0, 1], [1, 1]]), np.array([[0, 0], [1, 0], [0, 1], [2, 2]]), 2/3, "curated_example_2")
curated_example_3 = Example(np.array([[0, 0], [2, 0], [3, 0]]), np.array([[0, 0], [2, 0], [1, 0]]), 1/3, "curated_example_3")

def test_tsi_on_example(example):
    X = example.X
    Y = example.Y
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    tsi = TSI()
    assert tsi(X, Y, d_x, d_y) == example.expected_tsi
    print(f"TSI on example {example.name} test passed")

def test_tsi_on_random_data():
    X = np.random.rand(30, 3)
    Y = X
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    tsi = TSI()
    assert tsi(X, Y, d_x, d_y) == 1.0
    print("TSI on random data test passed")

def test_approx_tsi_alignment_with_tsi():
    X = np.random.rand(30, 3)
    Y = np.random.rand(30, 3)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    approx_tsi = ApproxTSI(k=20000)
    tsi = TSI()
    assert np.abs(approx_tsi(X, Y, d_x, d_y) - tsi(X, Y, d_x, d_y)) < 1e-2
    print("ApproxTSI alignment with TSI test passed")

def test_efficient_tsi_on_example(example):
    X = example.X
    Y = example.Y
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    efficient_tsi = EfficientTSI()
    assert efficient_tsi(X, Y, d_x, d_y) == example.expected_tsi
    print(f"EfficientTSI on example {example.name} test passed")

def test_efficient_tsi_alignment_with_tsi_on_random_data():
    X = np.random.rand(30, 3)
    Y = np.random.rand(30, 3)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    efficient_tsi = EfficientTSI()
    tsi = TSI()
    assert efficient_tsi(X, Y, d_x, d_y) == tsi(X, Y, d_x, d_y)
    print("EfficientTSI alignment with TSI on random data test passed")

def test_efficient_tsi_alignment_with_tsi_on_random_data_with_equalities():
    X = np.random.rand(30, 3)
    Y = np.concatenate((X[:15], X[:15]), axis=0)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    efficient_tsi = EfficientTSI()
    tsi = TSI()
    assert efficient_tsi(X, Y, d_x, d_y) == tsi(X, Y, d_x, d_y)
    print("EfficientTSI alignment with TSI test on random data with equalities passed")

def compare_tsi_and_efficient_tsi():
    X = np.random.rand(50, 20)
    Y = np.random.rand(50, 20)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    efficient_tsi = EfficientTSI()
    tsi = TSI()
    start_time = time.time()
    tsi(X, Y, d_x, d_y)
    end_time = time.time()
    print(f"Time taken by TSI: {end_time - start_time}")
    start_time = time.time()
    efficient_tsi(X, Y, d_x, d_y)
    end_time = time.time()
    print(f"Time taken by EfficientTSI: {end_time - start_time}")

if __name__ == "__main__":
    test_tsi_on_example(curated_example_1)
    test_tsi_on_example(curated_example_2)
    test_tsi_on_example(curated_example_3)
    test_tsi_on_random_data()
    test_approx_tsi_alignment_with_tsi()
    test_efficient_tsi_on_example(curated_example_1)
    test_efficient_tsi_on_example(curated_example_3)
    test_efficient_tsi_alignment_with_tsi_on_random_data()
    test_efficient_tsi_alignment_with_tsi_on_random_data_with_equalities()
    compare_tsi_and_efficient_tsi()