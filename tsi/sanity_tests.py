#!/usr/bin/env python3

import time
import argparse
from tsi import TSI, ApproxTSI, EfficientTSI, RepresentationPair, PartialIndices, CompleteIndices
import numpy as np

class Example:
    def __init__(self, representations: RepresentationPair, expected_tsi, name, indices=None):
        self.name = name
        self.representations = representations
        self.expected_tsi = expected_tsi
        self.indices = indices

d_x = lambda x, y: np.linalg.norm(x - y)
d_y = lambda x, y: np.linalg.norm(x - y)

curated_example_1 = Example(RepresentationPair(X=np.array([[0, 0], [1, 0], [0, 1]]), Y=np.array([[0, 0], [1, 0], [1, 1]]), d_x=d_x, d_y=d_y), 0.0, "curated_example_1")
curated_example_2 = Example(RepresentationPair(X=np.array([[0, 0], [1, 0], [0, 1], [1, 1]]), Y=np.array([[0, 0], [1, 0], [0, 1], [2, 2]]), d_x=d_x, d_y=d_y), 2/3, "curated_example_2")
curated_example_3 = Example(RepresentationPair(X=np.array([[0, 0], [2, 0], [3, 0]]), Y=np.array([[0, 0], [2, 0], [1, 0]]), d_x=d_x, d_y=d_y), 1/3, "curated_example_3")

curated_example_4_with_partial_indices = Example(RepresentationPair(X=np.array([[0, 0], [1, 0], [0, 1]]), Y=np.array([[0, 0], [1, 0], [1, 1]]), d_x=d_x, d_y=d_y), 0.0, "curated_example_4", indices=PartialIndices(indices=[(0, 1, 2)]))
curated_example_5_with_partial_indices = Example(RepresentationPair(X=np.array([[0, 0], [2, 0], [3, 0]]), Y=np.array([[0, 0], [2, 0], [1, 0]]), d_x=d_x, d_y=d_y), 1.0, "curated_example_5", indices=PartialIndices(indices=[(1, 2, 0)]))

curated_example_6_with_complete_indices = Example(RepresentationPair(X=np.array([[0, 0], [1, 0], [0, 1]]), Y=np.array([[0, 0], [1, 0], [1, 1]]), d_x=d_x, d_y=d_y), 0.0, "curated_example_6", indices=CompleteIndices(indices={0: [1, 2], 1: [0, 2]}))
curated_example_7_with_complete_indices = Example(RepresentationPair(X=np.array([[0, 0], [1, 0], [0, 1], [1, 1]]), Y=np.array([[0, 0], [1, 0], [0, 1], [2, 2]]), d_x=d_x, d_y=d_y), 1/2, "curated_example_7", indices=CompleteIndices(indices={0: [1, 2], 1: [0, 2, 3]}))


def test_tsi_on_example(example):
    tsi = TSI()
    assert tsi(example.representations) == example.expected_tsi
    print(f"TSI on example {example.name} test passed")

def test_approx_tsi_on_example_with_partial_indices(example):
    approx_tsi = ApproxTSI()
    assert approx_tsi(example.representations, example.indices) == example.expected_tsi
    print(f"ApproxTSI on example {example.name} test with partial indices passed")

def test_approx_tsi_on_example_with_complete_indices(example):
    approx_tsi = ApproxTSI()
    assert approx_tsi(example.representations, example.indices) == example.expected_tsi
    print(f"ApproxTSI on example {example.name} test with complete indices passed")

def test_tsi_on_random_data():
    X = np.random.rand(30, 3)
    Y = X
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    tsi = TSI()
    assert tsi(representations) == 1.0
    print("TSI on random data test passed")

def test_approx_tsi_alignment_with_tsi_with_partial_indices():
    X = np.random.rand(30, 3)
    Y = np.random.rand(30, 3)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    partial_indices = PartialIndices(indices=[(i,j,k) for i in range(30) for j in range(30) for k in range(30) if i != j and i != k and j != k])
    approx_tsi = ApproxTSI()
    tsi = TSI()
    assert approx_tsi(representations, partial_indices) == tsi(representations)
    print("ApproxTSI alignment with TSI test with partial indices passed")

def test_approx_tsi_alignment_with_tsi_with_complete_indices():
    X = np.random.rand(30, 3)
    Y = np.random.rand(30, 3)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    complete_indices = CompleteIndices(indices={i: [j for j in range(30) if j != i] for i in range(30)})
    approx_tsi = ApproxTSI()
    tsi = TSI()
    assert approx_tsi(representations, complete_indices) == tsi(representations)
    print("ApproxTSI alignment with TSI test with complete indices passed")

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
    assert efficient_tsi(representations) == tsi(representations)
    print("EfficientTSI alignment with TSI on random data test passed")

def test_efficient_tsi_alignment_with_tsi_on_random_data_with_equalities():
    X = np.random.rand(30, 3)
    Y = np.concatenate((X[:15], X[:15]), axis=0)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    efficient_tsi = EfficientTSI(euclidean=False)
    tsi = TSI()
    assert efficient_tsi(representations) == tsi(representations)
    print("EfficientTSI alignment with TSI test on random data with equalities passed")

def compare_tsi_and_efficient_tsi():
    X = np.random.rand(50, 20)
    Y = np.random.rand(50, 20)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    efficient_tsi = EfficientTSI(euclidean=False)
    tsi = TSI()
    start_time = time.time()
    tsi(representations)
    end_time = time.time()
    print(f"Time taken by TSI: {end_time - start_time}")
    start_time = time.time()
    efficient_tsi(representations)
    end_time = time.time()
    print(f"Time taken by EfficientTSI: {end_time - start_time}")

def run_tsi_tests():
    """Run tests for TSI implementation"""
    test_tsi_on_example(curated_example_1)
    test_tsi_on_example(curated_example_2)
    test_tsi_on_example(curated_example_3)
    test_tsi_on_random_data()

def run_approx_tsi_tests():
    """Run tests for ApproxTSI implementation"""
    test_approx_tsi_on_example_with_partial_indices(curated_example_4_with_partial_indices)
    test_approx_tsi_on_example_with_partial_indices(curated_example_5_with_partial_indices)
    test_approx_tsi_on_example_with_complete_indices(curated_example_6_with_complete_indices)
    test_approx_tsi_on_example_with_complete_indices(curated_example_7_with_complete_indices)
    test_approx_tsi_alignment_with_tsi_with_partial_indices()
    test_approx_tsi_alignment_with_tsi_with_complete_indices()

def run_efficient_tsi_tests():
    """Run tests for EfficientTSI implementation"""
    test_efficient_tsi_on_example(curated_example_1)
    test_efficient_tsi_on_example(curated_example_2)
    test_efficient_tsi_on_example(curated_example_3)
    test_efficient_tsi_alignment_with_tsi_on_random_data()
    test_efficient_tsi_alignment_with_tsi_on_random_data_with_equalities()

def main():
    parser = argparse.ArgumentParser(description='Run TSI sanity tests')
    parser.add_argument('--test-subject', 
                       choices=['TSI', 'ApproxTSI', 'EfficientTSI'], 
                       required=True,
                       help='Specify which TSI implementation to test')
    
    args = parser.parse_args()
    
    print(f"Running tests for {args.test_subject}")
    
    if args.test_subject == 'TSI':
        run_tsi_tests()
    elif args.test_subject == 'ApproxTSI':
        run_approx_tsi_tests()
    elif args.test_subject == 'EfficientTSI':
        run_efficient_tsi_tests()
    
    # Always run the comparison at the end if EfficientTSI is being tested
    if args.test_subject == 'EfficientTSI':
        print("\nRunning performance comparison:")
        compare_tsi_and_efficient_tsi()
    
    print(f"\nAll {args.test_subject} tests completed successfully!")

if __name__ == "__main__":
    main()