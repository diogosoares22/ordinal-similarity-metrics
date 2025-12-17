import time
from scipy.optimize import linprog
import numpy as np
import itertools
import math
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from src.tsi import TSI
from src.qsi import QSI
from src.data import RepresentationPair


def index_to_variables(i: int, j: int, n: int) -> int:
    return i * (n - 2) + j - 1 - (i * (i - 1)) // 2


def get_permutations_with_networkx(n, constraints):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(constraints)

    for topo_sort in nx.all_topological_sorts(G):
        result = [0] * n
        for rank, index in enumerate(topo_sort, start=1):
            result[rank - 1] = index

        yield result


def distances_to_X(distances: np.ndarray, n: int) -> list:
    X = [0.0]
    for i in range(1, n):
        X.append(distances[i - 1])
    return X


def absolute_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute absolute distance between two arrays (picklable for multiprocessing)."""
    return np.abs(x - y)


def create_metric_instance(metric_name: str):
    """Create a metric instance from its name."""
    if metric_name == "tsi":
        return TSI()
    elif metric_name == "qsi":
        return QSI()
    else:
        raise ValueError(f"Invalid metric: {metric_name}")


def check_x_has_equivalent(
    index: int,
    all_Xs: list,
    metric_name: str,
) -> bool:
    """
    Check if X at index has an equivalent X (metric score = 1) among later Xs.

    Returns True if an equivalent exists (should be excluded), False otherwise.
    """
    metric = create_metric_instance(metric_name)
    x_reference = all_Xs[index]

    for x_target in all_Xs[index + 1:]:
        representation_pair = RepresentationPair(
            np.array([x_reference]).reshape(-1, 1),
            np.array([x_target]).reshape(-1, 1),
            absolute_distance,
            absolute_distance,
        )
        if metric(representation_pair) == 1:
            return True
    return False


def remove_all_metric_invariant_Xs(
    Xs: list,
    metric: str = "tsi",
    max_workers: int | None = None,
) -> list:
    """
    Remove Xs that are metric-equivalent to later Xs in the list.

    Args:
        Xs: List of X configurations
        metric: Metric name ("tsi" or "qsi")
        max_workers: Maximum number of parallel workers (None = number of CPUs)

    Returns:
        List of structurally unique X configurations
    """
    if len(Xs) == 0:
        return []

    # Convert to list of arrays for consistent handling
    Xs_arrays = [np.array(x) if not isinstance(x, np.ndarray) else x for x in Xs]

    # Create partial function with fixed arguments
    check_fn = partial(
        check_x_has_equivalent,
        all_Xs=Xs_arrays,
        metric_name=metric,
    )

    # Check each X in parallel
    indices = range(len(Xs_arrays))
    num_tasks = len(indices)
    has_equivalent = [False] * num_tasks
    
    # Pre-calculate total work (number of pairs to check)
    # Task i checks (num_tasks - 1 - i) pairs
    total_work = sum(num_tasks - 1 - i for i in indices)
    work_done = 0
    
    print(f"Starting metric-invariant removal for {num_tasks} configurations...")
    start_time = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(check_fn, i): i for i in indices}
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
                has_equivalent[idx] = result
                completed += 1
                
                # Update work done based on the index that finished
                work_done += (num_tasks - 1 - idx)
                
                elapsed = time.time() - start_time
                if work_done > 0:
                    avg_time_per_work = elapsed / work_done
                    remaining_work = total_work - work_done
                    est_time_left = avg_time_per_work * remaining_work
                else:
                    est_time_left = 0 # Or some default
                
                print(f"\rInvariant removal: {completed}/{num_tasks} complete. "
                      f"Elapsed: {elapsed:.1f}s, Est. left: {est_time_left:.1f}s", end="", flush=True)
            except Exception as exc:
                print(f"\nIndex {idx} generated an exception: {exc}")

    print(f"\nCompleted invariant removal in {time.time() - start_time:.1f}s.")

    # Filter: keep only Xs that don't have an equivalent
    structural_unique_Xs = [
        Xs[i] for i, has_eq in enumerate(has_equivalent) if not has_eq
    ]

    return structural_unique_Xs


def adjust_to_nearest_eps(Xs: list, eps: float) -> list:
    return [np.round(np.array(x) / eps) * eps for x in Xs]


def build_triangle_equality_constraints(n: int, variables: int) -> tuple[list, list]:
    """Build distance triangle equality constraints for the LP."""
    A_eq = []
    b_eq = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                constraint_row = [0] * variables
                constraint_row[index_to_variables(i, j, n)] = 1
                constraint_row[index_to_variables(j, k, n)] = 1
                constraint_row[index_to_variables(i, k, n)] = -1
                A_eq.append(constraint_row)
                b_eq.append(0)
    return A_eq, b_eq


def build_permutation_constraints(n: int, variables: int) -> list[tuple[int, int]]:
    """Build permutation constraints as pairs of (pivot_index, compare_index)."""
    permutation_constraints = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            pivot_index = index_to_variables(i, j, n)
            for k in range(j + 1, n):
                compare_index = index_to_variables(i, k, n)
                permutation_constraints.append((pivot_index, compare_index))
            for t in range(i):
                compare_index = index_to_variables(t, j, n)
                permutation_constraints.append((pivot_index, compare_index))
    return permutation_constraints


def solve_single_operator_configuration(
    operator_index: int,
    ordering: tuple,
    variables: int,
    eps: float,
    base_A_eq: list,
    base_b_eq: list,
) -> np.ndarray | None:
    """Solve LP for a single operator configuration (equality vs inequality pattern)."""
    c = [0] * variables
    bounds = (eps, None)

    specific_A_ub = []
    specific_b_ub = []
    specific_A_eq = base_A_eq.copy()
    specific_b_eq = base_b_eq.copy()

    binary_flips = [int(b) for b in format(operator_index, f"0{variables - 1}b")]

    for idx, binary_flip in enumerate(binary_flips):
        constraint_row = [0] * variables
        constraint_row[ordering[idx]] = 1
        constraint_row[ordering[idx + 1]] = -1
        if binary_flip == 1:  # Strict inequality
            specific_A_ub.append(constraint_row)
            specific_b_ub.append(-eps)
        else:  # Equality constraint
            specific_A_eq.append(constraint_row)
            specific_b_eq.append(0)

    # Solve LP based on which constraints are present
    if len(specific_A_ub) == 0:
        res = linprog(c, A_eq=specific_A_eq, b_eq=specific_b_eq, bounds=bounds, method="highs")
    elif len(specific_A_eq) == 0:
        res = linprog(c, A_ub=specific_A_ub, b_ub=specific_b_ub, bounds=bounds, method="highs")
    else:
        res = linprog(
            c,
            A_ub=specific_A_ub,
            b_ub=specific_b_ub,
            A_eq=specific_A_eq,
            b_eq=specific_b_eq,
            bounds=bounds,
            method="highs",
        )

    if res.status == 0:
        return res.x
    return None


def process_single_ordering(
    ordering: tuple,
    variables: int,
    eps: float,
    base_A_eq: list,
    base_b_eq: list,
) -> list[np.ndarray]:
    """Process all operator configurations for a single ordering."""
    valid_distances = []
    num_operator_configs = 2 ** (variables - 1)

    for operator_index in range(num_operator_configs):
        result = solve_single_operator_configuration(
            operator_index, ordering, variables, eps, base_A_eq, base_b_eq
        )
        if result is not None:
            valid_distances.append(result)

    return valid_distances


def get_all_representative_valid_Xs(
    n: int,
    eps: float = 1e-6,
    use_networkx: bool = True,
    metric: str = "tsi",
    max_workers: int | None = None,
) -> list:
    """
    Find all representative valid X configurations.

    Args:
        n: Number of points
        eps: Epsilon for numerical precision
        use_networkx: Whether to use networkx for topological sorting
        metric: Metric to use for removing invariant Xs ("tsi" or "qsi")
        max_workers: Maximum number of parallel workers (None = number of CPUs)

    Returns:
        List of representative unique X configurations
    """
    variables = (n * (n - 1)) // 2

    # Build base constraints
    base_A_eq, base_b_eq = build_triangle_equality_constraints(n, variables)
    permutation_constraints = build_permutation_constraints(n, variables)

    # Get orderings iterator
    if use_networkx:
        orderings = list(get_permutations_with_networkx(variables, permutation_constraints))
    else:
        orderings = list(itertools.permutations(range(variables)))

    # Convert orderings to tuples for pickling in multiprocessing
    orderings = [tuple(o) for o in orderings]

    # Create partial function with fixed arguments
    process_ordering_fn = partial(
        process_single_ordering,
        variables=variables,
        eps=eps,
        base_A_eq=base_A_eq,
        base_b_eq=base_b_eq,
    )

    # Process orderings in parallel
    valid_distances = []
    num_orderings = len(orderings)
    print(f"Processing {num_orderings} orderings...")
    start_time = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_ordering_fn, o) for o in orderings]
        for future in as_completed(futures):
            try:
                ordering_results = future.result()
                valid_distances.extend(ordering_results)
                completed += 1
                
                elapsed = time.time() - start_time
                avg_time = elapsed / completed
                remaining = num_orderings - completed
                est_time_left = avg_time * remaining
                
                print(f"\rProcess Orderings: {completed}/{num_orderings} complete. "
                      f"Elapsed: {elapsed:.1f}s, Est. left: {est_time_left:.1f}s", end="", flush=True)
            except Exception as exc:
                print(f"\nAn ordering generated an exception: {exc}")

    print(f"\nCompleted processing orderings in {time.time() - start_time:.1f}s.")

    # Convert distances to X configurations and filter
    representative_valid_Xs = [
        distances_to_X(distance_vector, n) for distance_vector in valid_distances
    ]
    adjusted_representative_valid_Xs = adjust_to_nearest_eps(representative_valid_Xs, eps)
    representative_unique_Xs = remove_all_metric_invariant_Xs(
        adjusted_representative_valid_Xs, metric, max_workers=max_workers
    )

    return representative_unique_Xs