#!/usr/bin/env python3

import time
import argparse
from tsi import TSI, ApproxTSI, EfficientTSI, RepresentationPair, PartialIndices, CompleteIndices, NearestNeighborTSI, BatchTSI, OddOneOutTSI
import numpy as np
from dataclasses import dataclass

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

### TSI tests ###

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

### ApproxTSI tests ###

curated_example_1_partial_indices = Example(RepresentationPair(X=np.array([[0, 0], [1, 0], [0, 1]]), Y=np.array([[0, 0], [1, 0], [1, 1]]), d_x=d_x, d_y=d_y), 0.0, "curated_example_1_partial_indices", indices=PartialIndices(indices=[(0, 1, 2)]))
curated_example_2_partial_indices = Example(RepresentationPair(X=np.array([[0, 0], [2, 0], [3, 0]]), Y=np.array([[0, 0], [2, 0], [1, 0]]), d_x=d_x, d_y=d_y), 1.0, "curated_example_2_partial_indices", indices=PartialIndices(indices=[(1, 2, 0)]))

curated_example_1_complete_indices = Example(RepresentationPair(X=np.array([[0, 0], [1, 0], [0, 1]]), Y=np.array([[0, 0], [1, 0], [1, 1]]), d_x=d_x, d_y=d_y), 0.0, "curated_example_1_complete_indices", indices=CompleteIndices(indices={0: [1, 2], 1: [0, 2]}))
curated_example_2_complete_indices = Example(RepresentationPair(X=np.array([[0, 0], [1, 0], [0, 1], [1, 1]]), Y=np.array([[0, 0], [1, 0], [0, 1], [2, 2]]), d_x=d_x, d_y=d_y), 1/2, "curated_example_2_complete_indices", indices=CompleteIndices(indices={0: [1, 2], 1: [0, 2, 3]}))


def test_approx_tsi_on_example_with_partial_indices(example):
    approx_tsi = ApproxTSI()
    assert approx_tsi(example.representations, example.indices) == example.expected_tsi
    print(f"ApproxTSI on example {example.name} test with partial indices passed")

def test_approx_tsi_on_example_with_complete_indices(example):
    approx_tsi = ApproxTSI()
    assert approx_tsi(example.representations, example.indices) == example.expected_tsi
    print(f"ApproxTSI on example {example.name} test with complete indices passed")

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

### NearestNeighborTSI tests ###

curated_example_1_2_nearest_neighbor_tsi = Example(RepresentationPair(X=np.array([[0, 0], [1, 0], [0, 1], [5, 0], [7, 0], [8, 0]]), Y=np.array([[0, 0], [1, 0], [1, 1], [5, 0], [7, 0], [6, 0]]), d_x=d_x, d_y=d_y), 1/6, "curated_example_1_nearest_neighbor_tsi")
curated_example_2_2_nearest_neighbor_tsi = Example(RepresentationPair(X=np.array([[0, 0], [1, 0], [0, 1], [5, 0], [7, 0], [8, 0]]), Y=np.array([[0, 0], [1, 0], [0, 1], [5, 0], [7, 0], [6, 0]]), d_x=d_x, d_y=d_y), 2/3, "curated_example_2_nearest_neighbor_tsi")
curated_example_3_2_nearest_neighbor_tsi = Example(RepresentationPair(X=np.array([[0, 0], [1, 0], [0, 1], [5, 0], [7, 0], [8, 0]]), Y=np.array([[0, 0], [1, 0], [0, 1], [5, 0], [7, 0], [8, 0]]), d_x=d_x, d_y=d_y), 1, "curated_example_3_nearest_neighbor_tsi")
curated_example_4_3_nearest_neighbor_tsi = Example(RepresentationPair(X=np.array([[0, 0], [1, 0], [0, 1], [1, 1]]), Y=np.array([[0, 0], [1, 0], [0, 1], [2, 2]]), d_x=d_x, d_y=d_y), 2/3, "curated_example_4_nearest_neighbor_tsi")
curated_example_5_2_nearest_neighbor_tsi = Example(RepresentationPair(X=np.array([[0, 0], [1, 0], [0, 1], [3, 3]]), Y=np.array([[0, 0], [1, 0], [0, 1], [-0.1, -0.1]]), d_x=d_x, d_y=d_y), 1/2, "curated_example_5_nearest_neighbor_tsi")


def test_nearest_neighbor_tsi_on_example(example, k):
    nearest_neighbor_tsi = NearestNeighborTSI(k=k, euclidean=False)
    assert nearest_neighbor_tsi(example.representations) == example.expected_tsi
    print(f"NearestNeighborTSI on example {example.name} test passed")

def test_nearest_neighbor_tsi_alignment_with_tsi_on_random_data():
    X = np.random.rand(30, 3)
    Y = np.random.rand(30, 3)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    nearest_neighbor_tsi = NearestNeighborTSI(k=29, euclidean=False)
    tsi = TSI()
    assert nearest_neighbor_tsi(representations) == tsi(representations)
    print("NearestNeighborTSI alignment with TSI on random data test passed")

def test_nearest_neighbor_tsi_alignment_with_tsi_on_random_data_with_equalities():
    X = np.random.rand(30, 3)
    Y = np.concatenate((X[:15], X[:15]), axis=0)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    nearest_neighbor_tsi = NearestNeighborTSI(k=29, euclidean=False)
    tsi = TSI()
    assert nearest_neighbor_tsi(representations) == tsi(representations)
    print("NearestNeighborTSI alignment with TSI on random data with equalities test passed")

### BatchTSI tests ###

curated_example_1_3_batch_tsi = Example(RepresentationPair(X=np.array([[0, 0], [1, 0], [0, 1], [5, 0], [7, 0], [8, 0]]), Y=np.array([[0, 0], [1, 0], [1, 1], [5, 0], [7, 0], [6, 0]]), d_x=d_x, d_y=d_y), 1/6, "curated_example_1_batch_tsi")
curated_example_2_3_batch_tsi = Example(RepresentationPair(X=np.array([[0, 0], [2, 0], [3, 0], [5, 0], [7, 0], [8, 0]]), Y=np.array([[0, 0], [2, 0], [3, 0], [5, 0], [7, 0], [6, 0]]), d_x=d_x, d_y=d_y), 2/3, "curated_example_2_batch_tsi")
curated_example_3_3_batch_tsi = Example(RepresentationPair(X=np.array([[0, 0], [2, 0], [3, 0], [5, 0], [7, 0], [8, 0], [-1, -1]]), Y=np.array([[0, 0], [2, 0], [3, 0], [5, 0], [7, 0], [6, 0], [-1, -1]]), d_x=d_x, d_y=d_y), 2/3, "curated_example_3_batch_tsi")


def test_batch_tsi_on_example(example):
    batch_tsi = BatchTSI(euclidean=False, batch_size=3)
    assert batch_tsi(example.representations) == example.expected_tsi
    print(f"BatchTSI on example {example.name} test passed")

def test_batch_tsi_alignment_with_tsi_on_random_data():
    X = np.random.rand(30, 3)
    Y = np.random.rand(30, 3)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    batch_tsi = BatchTSI(euclidean=False, batch_size=30)
    tsi = TSI()
    assert batch_tsi(representations) == tsi(representations)
    print("BatchTSI alignment with TSI on random data test passed")

def test_batch_tsi_alignment_with_tsi_on_random_data_with_equalities():
    X = np.random.rand(30, 3)
    Y = np.concatenate((X[:15], X[:15]), axis=0)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    batch_tsi = BatchTSI(euclidean=False, batch_size=30)
    tsi = TSI()
    assert batch_tsi(representations) == tsi(representations)
    print("BatchTSI alignment with TSI on random data with equalities test passed")


### OddOneOutTSI tests ###

class OddOneOutExample:
    def __init__(self, X, d_x, odd_one_out_observations, expected_tsi, name):
        self.name = name
        self.X = X
        self.d_x = d_x
        self.odd_one_out_observations = odd_one_out_observations
        self.expected_tsi = expected_tsi


curated_example_1_odd_one_out_tsi = OddOneOutExample(X=np.array([[0, 0], [1, 0], [3, 0], [4, 0]]), d_x=d_x, odd_one_out_observations={(0, 1, 2): 2, (0, 1, 3): 3, (0, 2, 3): 0}, expected_tsi=1, name="curated_example_1_odd_one_out_tsi")
curated_example_2_odd_one_out_tsi = OddOneOutExample(X=np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]), d_x=d_x, odd_one_out_observations={(0, 1, 3): 1, (0, 1, 4): 0}, expected_tsi=1/4, name="curated_example_2_odd_one_out_tsi")

def test_odd_one_out_tsi_on_example(odd_one_out_example):
    odd_one_out_tsi = OddOneOutTSI(odd_one_out_observations=odd_one_out_example.odd_one_out_observations)
    assert odd_one_out_tsi(odd_one_out_example.X, odd_one_out_example.d_x) == odd_one_out_example.expected_tsi
    print(f"OddOneOutTSI on example {odd_one_out_example.name} test passed")

def run_tsi_tests():
    """Run tests for TSI implementation"""
    test_tsi_on_example(curated_example_1)
    test_tsi_on_example(curated_example_2)
    test_tsi_on_example(curated_example_3)
    test_tsi_on_random_data()

def run_approx_tsi_tests():
    """Run tests for ApproxTSI implementation"""
    test_approx_tsi_on_example_with_partial_indices(curated_example_1_partial_indices)
    test_approx_tsi_on_example_with_partial_indices(curated_example_2_partial_indices)
    test_approx_tsi_alignment_with_tsi_with_partial_indices()
    test_approx_tsi_on_example_with_complete_indices(curated_example_1_complete_indices)
    test_approx_tsi_on_example_with_complete_indices(curated_example_2_complete_indices)
    test_approx_tsi_alignment_with_tsi_with_complete_indices()

def run_efficient_tsi_tests():
    """Run tests for EfficientTSI implementation"""
    test_efficient_tsi_on_example(curated_example_1)
    test_efficient_tsi_on_example(curated_example_2)
    test_efficient_tsi_on_example(curated_example_3)
    test_efficient_tsi_alignment_with_tsi_on_random_data()
    test_efficient_tsi_alignment_with_tsi_on_random_data_with_equalities()

def run_nearest_neighbor_tsi_tests():
    """Run tests for NearestNeighborTSI implementation"""
    test_nearest_neighbor_tsi_on_example(curated_example_1_2_nearest_neighbor_tsi, 2)
    test_nearest_neighbor_tsi_on_example(curated_example_2_2_nearest_neighbor_tsi, 2)
    test_nearest_neighbor_tsi_on_example(curated_example_3_2_nearest_neighbor_tsi, 2)
    test_nearest_neighbor_tsi_on_example(curated_example_4_3_nearest_neighbor_tsi, 3)
    test_nearest_neighbor_tsi_on_example(curated_example_5_2_nearest_neighbor_tsi, 2)
    test_nearest_neighbor_tsi_alignment_with_tsi_on_random_data()
    test_nearest_neighbor_tsi_alignment_with_tsi_on_random_data_with_equalities()

def run_batch_tsi_tests():
    """Run tests for BatchTSI implementation"""
    test_batch_tsi_on_example(curated_example_1_3_batch_tsi)
    test_batch_tsi_on_example(curated_example_2_3_batch_tsi)
    test_batch_tsi_on_example(curated_example_3_3_batch_tsi)
    test_batch_tsi_alignment_with_tsi_on_random_data()
    test_batch_tsi_alignment_with_tsi_on_random_data_with_equalities()

def run_odd_one_out_tsi_tests():
    """Run tests for OddOneOutTSI implementation"""
    test_odd_one_out_tsi_on_example(curated_example_1_odd_one_out_tsi)
    test_odd_one_out_tsi_on_example(curated_example_2_odd_one_out_tsi)


def main():
    parser = argparse.ArgumentParser(description='Run TSI sanity tests')
    parser.add_argument('--test-subject', 
                       choices=['All', 'TSI', 'ApproxTSI', 'EfficientTSI', 'NearestNeighborTSI', 'BatchTSI', 'OddOneOutTSI'], 
                       default='All',
                       help='Specify which TSI implementation to test')
    
    args = parser.parse_args()
    
    print(f"Running tests for {args.test_subject}")

    
    if args.test_subject == 'All':
        run_tsi_tests()
        run_approx_tsi_tests()
        run_efficient_tsi_tests()
        run_nearest_neighbor_tsi_tests()
        run_batch_tsi_tests()
        run_odd_one_out_tsi_tests()
    elif args.test_subject == 'TSI':
        run_tsi_tests()
    elif args.test_subject == 'ApproxTSI':
        run_approx_tsi_tests()
    elif args.test_subject == 'EfficientTSI':
        run_efficient_tsi_tests()
    elif args.test_subject == 'NearestNeighborTSI':
        run_nearest_neighbor_tsi_tests()
    elif args.test_subject == 'BatchTSI':
        run_batch_tsi_tests()
    elif args.test_subject == 'OddOneOutTSI':
        run_odd_one_out_tsi_tests()
    
    print(f"\nAll {args.test_subject} tests completed successfully!")

if __name__ == "__main__":
    main()