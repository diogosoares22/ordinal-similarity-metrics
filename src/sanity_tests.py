#!/usr/bin/env python3

import argparse
from src.data import RepresentationPair
from src.tsi import TSI, EfficientTSI, ApproxTSI, EfficientApproxTSI
from src.qsi import QSI, ApproxQSI, EfficientQSI, EfficientApproxQSI
import numpy as np

EPSILON = 0.01
DELTA = 0.001

class Example:
    def __init__(self, representations: RepresentationPair, expected_tsi, expected_qsi, name, indices=None):
        self.name = name
        self.representations = representations
        self.expected_tsi = expected_tsi
        self.expected_qsi = expected_qsi
        self.indices = indices

d_x = lambda x, y: np.linalg.norm(x - y)
d_y = lambda x, y: np.linalg.norm(x - y)

curated_example_1 = Example(RepresentationPair(X=np.array([[0, 0], [1, 0], [0, 1]]), Y=np.array([[0, 0], [1, 0], [1, 1]]), d_x=d_x, d_y=d_y), 0.0, None, "curated_example_1")
curated_example_2 = Example(RepresentationPair(X=np.array([[0, 0], [1, 0], [0, 1], [1, 1]]), Y=np.array([[0, 0], [1, 0], [0, 1], [2, 2]]), d_x=d_x, d_y=d_y), 2/3, 0, "curated_example_2")
curated_example_3 = Example(RepresentationPair(X=np.array([[0, 0], [2, 0], [3, 0]]), Y=np.array([[0, 0], [2, 0], [1, 0]]), d_x=d_x, d_y=d_y), 1/3, None,"curated_example_3")

### TSI tests ###

def test_tsi_on_example(example):
    tsi = TSI()
    assert tsi(example.representations) == example.expected_tsi
    print(f"TSI on example {example.name} test passed")

def test_tsi_on_identical_data():
    X = np.random.rand(30, 3)
    Y = X
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    tsi = TSI()
    assert tsi(representations) == 1.0
    print("TSI on identical data test passed")

### EfficientTSI tests ###

def test_efficient_tsi_on_example(example):
    efficient_tsi = EfficientTSI(euclidean=False)
    assert efficient_tsi(example.representations) == example.expected_tsi
    print(f"EfficientTSI on example {example.name} test passed")

def test_efficient_tsi_alignment_with_tsi_on_random_data():
    X = np.random.rand(30, 3)
    Y = np.random.rand(30, 3)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    efficient_tsi = EfficientTSI(euclidean=False)
    tsi = TSI()
    assert np.abs(efficient_tsi(representations) - tsi(representations)) <= 0.000001
    print("EfficientTSI alignment with TSI on random data test passed")

def test_efficient_tsi_alignment_with_tsi_on_random_data_with_equalities():
    X = np.random.rand(30, 3)
    Y = np.concatenate((X[:15], X[:15]), axis=0)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    efficient_tsi = EfficientTSI(euclidean=False)
    tsi = TSI()
    assert np.abs(efficient_tsi(representations) - tsi(representations)) <= 0.000001
    print("EfficientTSI alignment with TSI test on random data with equalities passed")

### ApproxTSI tests ###

def test_approx_tsi_on_example(example):
    approx_tsi = ApproxTSI(epsilon=EPSILON, delta=DELTA)
    value = approx_tsi(example.representations)
    assert abs(value - example.expected_tsi) <= EPSILON * 3 # 3 is a tolerance factor
    print(f"ApproxTSI on example {example.name} within tolerance test passed")

def test_approx_tsi_alignment_with_tsi_on_random_data():
    X = np.random.rand(30, 3)
    Y = np.random.rand(30, 3)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    approx_tsi = ApproxTSI(epsilon=EPSILON, delta=DELTA)
    tsi = TSI()
    approx_value = approx_tsi(representations)
    exact_value = tsi(representations)
    assert abs(approx_value - exact_value) <= EPSILON * 3 # 3 is a tolerance factor
    print("ApproxTSI alignment with TSI on random data within tolerance test passed")

### EfficientApproxTSI tests ###

def test_efficient_approx_tsi_on_example(example):
    efficient_approx_tsi = EfficientApproxTSI(seed=42)
    value = efficient_approx_tsi(example.representations)
    assert abs(value - example.expected_tsi) <= 0.1
    print(f"EfficientApproxTSI on example {example.name} within tolerance test passed")

def test_efficient_approx_tsi_alignment_with_tsi_on_random_data():
    X = np.random.rand(30, 3)
    Y = np.random.rand(30, 3)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    efficient_approx_tsi = EfficientApproxTSI(seed=42)
    tsi = TSI()
    approx_value = efficient_approx_tsi(representations)
    exact_value = tsi(representations)
    assert abs(approx_value - exact_value) <= 0.1
    print("EfficientApproxTSI alignment with TSI on random data within tolerance test passed")

### QSI tests ###

def test_qsi_on_example(example):
    qsi = QSI()
    assert qsi(example.representations) == example.expected_qsi
    print(f"QSI on example {example.name} test passed")

def test_qsi_on_identical_data():
    X = np.random.rand(20, 3)
    Y = X
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    qsi = QSI()
    assert qsi(representations) == 1.0
    print("QSI on identical data test passed")

### EfficientQSI tests ###

def test_efficient_qsi_on_example(example):
    efficient_qsi = EfficientQSI(euclidean=False)
    assert efficient_qsi(example.representations) == example.expected_qsi
    print(f"EfficientQSI on example {example.name} test passed")

def test_efficient_qsi_alignment_with_qsi_on_random_data():
    X = np.random.rand(20, 3)
    Y = np.random.rand(20, 3)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    efficient_qsi = EfficientQSI(euclidean=False)
    qsi = QSI()
    assert np.abs(efficient_qsi(representations) - qsi(representations)) <= 0.000001
    print("EfficientQSI alignment with QSI on random data test passed")

### ApproxQSI tests ###

def test_approx_qsi_on_example(example):
    approx_qsi = ApproxQSI(epsilon=EPSILON, delta=DELTA)
    value = approx_qsi(example.representations)
    assert abs(value - example.expected_qsi) <= EPSILON * 3 # 3 is a tolerance factor
    print(f"ApproxQSI on example {example.name} within tolerance test passed")

def test_approx_qsi_alignment_with_qsi_on_random_data():
    X = np.random.rand(20, 3)
    Y = np.random.rand(20, 3)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    qsi = QSI()
    approx_qsi = ApproxQSI(epsilon=EPSILON, delta=DELTA)
    exact_value = qsi(representations)
    approx_value = approx_qsi(representations)
    assert abs(approx_value - exact_value) <= EPSILON * 3 # 3 is a tolerance factor
    print("ApproxQSI alignment with QSI on random data within tolerance test passed")

### EfficientApproxQSI tests ###

def test_efficient_approx_qsi_on_example(example):
    efficient_approx_qsi = EfficientApproxQSI(seed=42)
    value = efficient_approx_qsi(example.representations)
    assert abs(value - example.expected_qsi) <= 0.1
    print(f"EfficientApproxQSI on example {example.name} within tolerance test passed")

def test_efficient_approx_qsi_alignment_with_qsi_on_random_data():
    X = np.random.rand(20, 3)
    Y = np.random.rand(20, 3)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    qsi = QSI()
    efficient_approx_qsi = EfficientApproxQSI(seed=42)
    exact_value = qsi(representations)
    approx_value = efficient_approx_qsi(representations)
    assert abs(approx_value - exact_value) <= 0.1
    print("EfficientApproxQSI alignment with QSI on random data within tolerance test passed")

def run_tsi_tests():
    """Run tests for TSI implementation"""
    test_tsi_on_example(curated_example_1)
    test_tsi_on_example(curated_example_2)
    test_tsi_on_example(curated_example_3)
    test_tsi_on_identical_data()

def run_efficient_tsi_tests():
    """Run tests for EfficientTSI implementation"""
    test_efficient_tsi_on_example(curated_example_1)
    test_efficient_tsi_on_example(curated_example_2)
    test_efficient_tsi_on_example(curated_example_3)
    test_efficient_tsi_alignment_with_tsi_on_random_data()
    test_efficient_tsi_alignment_with_tsi_on_random_data_with_equalities()

def run_approx_tsi_tests():
    """Run tests for ApproxTSI implementation"""
    test_approx_tsi_on_example(curated_example_1)
    test_approx_tsi_on_example(curated_example_2)
    test_approx_tsi_on_example(curated_example_3)
    test_approx_tsi_alignment_with_tsi_on_random_data()

def run_qsi_tests():
    """Run tests for QSI implementation"""
    test_qsi_on_example(curated_example_2)
    test_qsi_on_identical_data()

def run_efficient_qsi_tests():
    """Run tests for EfficientQSI implementation"""
    test_efficient_qsi_on_example(curated_example_2)
    test_efficient_qsi_alignment_with_qsi_on_random_data()

def run_approx_qsi_tests():
    """Run tests for ApproxQSI implementation"""
    test_approx_qsi_on_example(curated_example_2)
    test_approx_qsi_alignment_with_qsi_on_random_data()

def run_efficient_approx_tsi_tests():
    """Run tests for EfficientApproxTSI implementation"""
    test_efficient_approx_tsi_on_example(curated_example_1)
    test_efficient_approx_tsi_on_example(curated_example_2)
    test_efficient_approx_tsi_on_example(curated_example_3)
    test_efficient_approx_tsi_alignment_with_tsi_on_random_data()

def run_efficient_approx_qsi_tests():
    """Run tests for EfficientApproxQSI implementation"""
    test_efficient_approx_qsi_on_example(curated_example_2)
    test_efficient_approx_qsi_alignment_with_qsi_on_random_data()

def main():
    parser = argparse.ArgumentParser(description='Run TSI sanity tests')
    parser.add_argument('--test-subject', 
                       choices=['All', 'TSI', 'EfficientTSI', 'ApproxTSI', 'EfficientApproxTSI', 'QSI', 'ApproxQSI', 'EfficientQSI', 'EfficientApproxQSI'], 
                       default='All',
                       help='Specify which TSI implementation to test')
    
    args = parser.parse_args()
    
    print(f"Running tests for {args.test_subject}")

    
    if args.test_subject == 'All':
        run_tsi_tests()
        run_efficient_tsi_tests()
        run_approx_tsi_tests()
        run_efficient_approx_tsi_tests()
        run_qsi_tests()
        run_efficient_qsi_tests()
        run_approx_qsi_tests()
        run_efficient_approx_qsi_tests()
    elif args.test_subject == 'TSI':
        run_tsi_tests()
    elif args.test_subject == 'EfficientTSI':
        run_efficient_tsi_tests()
    elif args.test_subject == 'ApproxTSI':
        run_approx_tsi_tests()
    elif args.test_subject == 'EfficientApproxTSI':
        run_efficient_approx_tsi_tests()
    elif args.test_subject == 'QSI':
        run_qsi_tests()
    elif args.test_subject == 'ApproxQSI':
        run_approx_qsi_tests()
    elif args.test_subject == 'EfficientQSI':
        run_efficient_qsi_tests()
    elif args.test_subject == 'EfficientApproxQSI':
        run_efficient_approx_qsi_tests()
    
    print(f"\nAll {args.test_subject} tests completed successfully!")

if __name__ == "__main__":
    main()