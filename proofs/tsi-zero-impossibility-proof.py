from scipy.optimize import linprog
import numpy as np
import itertools
import ast


def create_inequality_for_triplet(x_i, x_j, x_k, i, j, k, d_x):
    d_x_ij = d_x(x_i, x_j)
    d_x_ik = d_x(x_i, x_k)
    coord_first = (min(i,j), max(i,j))
    coord_second = (min(i,k), max(i,k))
    if d_x_ij > d_x_ik:
        return [f"d_y_{coord_first}", "<=", f"d_y_{coord_second}"]
    elif d_x_ij == d_x_ik:
        return [f"d_y_{coord_first}", "!=", f"d_y_{coord_second}"]
    else:
        return [f"d_y_{coord_second}", "<=", f"d_y_{coord_first}"]


def get_all_constraints(X: list):
    """
    Generate all constraints from the input X.
    
    Returns:
        list: List of all inequality constraints
    """
    inequalities = []
    n = len(X)
    d_x = lambda x, y: np.abs(x - y)
    
    for i in range(n):
        for j in range(n):
            for k in range(j + 1, n):
                if i == j or i == k or j == k:
                    continue
                inequalities.append(create_inequality_for_triplet(X[i], X[j], X[k], i, j, k, d_x))
    
    return inequalities


def solver_with_constraints(constraints: list, n: int, eps: float = 1e-6):
    """
    Solve the LP feasibility problem with a given set of constraints.
    
    Args:
        constraints: List of inequality constraints to use
        n: Number of points (variables)
        eps: Small epsilon for strict inequalities
        
    Returns:
        scipy.optimize.OptimizeResult: The result of linprog
    """
    c = [0] * n
    num_vars = len(c)

    def parse_key(k):
        left, right = ast.literal_eval(k.replace("d_y_", ""))
        return left, right

    index_list = list(range(n))
    for ordering in itertools.permutations(index_list):
        A_ub = []
        b_ub = []
        
        # Ordering constraints: y[ordering[i]] < y[ordering[i+1]]
        for idx in range(n - 1):
            constraint_row = [0] * num_vars
            constraint_row[ordering[idx]] = 1
            constraint_row[ordering[idx + 1]] = -1
            A_ub.append(constraint_row)
            b_ub.append(-eps)

        diff_operators = []
        for row in constraints:
            lhs_left, lhs_right = parse_key(row[0])
            operator = row[1]
            rhs_left, rhs_right = parse_key(row[2])

            constraint_row = [0] * num_vars

            if operator == "<=":
                lhs_left_index = ordering.index(lhs_left)
                rhs_left_index = ordering.index(rhs_left)
                if lhs_left_index > rhs_left_index:
                    constraint_row[lhs_left_index] += 1
                    constraint_row[rhs_left_index] += -1
                else:
                    constraint_row[lhs_left_index] += -1
                    constraint_row[rhs_left_index] += 1

                lhs_right_index = ordering.index(lhs_right)
                rhs_right_index = ordering.index(rhs_right)
                if lhs_right_index > rhs_right_index:
                    constraint_row[lhs_right_index] += -1
                    constraint_row[rhs_right_index] += 1
                else:
                    constraint_row[lhs_right_index] += 1
                    constraint_row[rhs_right_index] += -1

                A_ub.append(constraint_row)
                b_ub.append(0)
            elif operator == "!=":
                diff_coord = (row[0], row[2])
                diff_operators.append(diff_coord)

        if len(diff_operators) > 0:
            for operator_index in range(2 ** len(diff_operators)):
                binary_flips = list(map(int, list(str(bin(operator_index))[2:])))
                specific_A_ub = A_ub.copy()
                specific_b_ub = b_ub.copy()
                for idx, binary_flip in enumerate(binary_flips):
                    constraint_row = [0] * num_vars
                    lhs_left, lhs_right = parse_key(diff_operators[idx][0])
                    rhs_left, rhs_right = parse_key(diff_operators[idx][1])
                    lhs_left_index = ordering.index(lhs_left)
                    lhs_right_index = ordering.index(lhs_right)
                    rhs_left_index = ordering.index(rhs_left)
                    rhs_right_index = ordering.index(rhs_right)
                    
                    def flip_sign(flip):
                        return 1 if flip == 1 else -1
                    
                    sign = flip_sign(binary_flip)
                    
                    if lhs_left_index > rhs_left_index:
                        constraint_row[lhs_left_index] += sign
                        constraint_row[rhs_left_index] += -sign
                    else:
                        constraint_row[lhs_left_index] += -sign
                        constraint_row[rhs_left_index] += sign
                    if lhs_right_index > rhs_right_index:
                        constraint_row[lhs_right_index] += -sign
                        constraint_row[rhs_right_index] += sign
                    else:
                        constraint_row[lhs_right_index] += sign
                        constraint_row[rhs_right_index] += -sign
                    specific_A_ub.append(constraint_row)
                    specific_b_ub.append(-eps)

                res = linprog(c, A_ub=specific_A_ub, b_ub=specific_b_ub, bounds=(None, None), method='highs')
                if res.status == 0:
                    return res
        else:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(None, None), method='highs')
            if res.status == 0:
                return res
    return res


def is_feasible_with_omissions(all_constraints: list, omitted_indices: set, n: int) -> bool:
    """
    Check if the system is feasible when omitting constraints at the given indices.
    
    Args:
        all_constraints: List of all constraints
        omitted_indices: Set of indices to omit
        n: Number of points
        
    Returns:
        bool: True if feasible, False otherwise
    """
    remaining_constraints = [c for i, c in enumerate(all_constraints) if i not in omitted_indices]
    result = solver_with_constraints(remaining_constraints, n)
    return result.status == 0


def tsi_exact_lower_bound(X: list, verbose: bool = True):
    """
    Find the exact minimum number of constraint omissions needed to make the system feasible.
    
    This performs an exhaustive search over all possible combinations of omissions,
    starting from 0 and incrementing until a feasible solution is found.
    
    Args:
        X: List of input values
        verbose: Whether to print progress
        
    Returns:
        tuple: (min_omissions, total_inequalities, ratio, best_omitted_indices)
    """
    n = len(X)
    all_constraints = get_all_constraints(X)
    total_inequalities = len(all_constraints)
    
    if verbose:
        print(f"Total number of inequalities: {total_inequalities}")
        print(f"Number of points: {n}")
    
    # Check if already feasible with 0 omissions
    if is_feasible_with_omissions(all_constraints, set(), n):
        if verbose:
            print("System is feasible with 0 omissions.")
        return 0, total_inequalities, 0.0, set()
    
    # Exhaustively search for minimum omissions
    for num_omissions in range(1, total_inequalities + 1):
        if verbose:
            num_combinations = np.math.comb(total_inequalities, num_omissions)
            print(f"Checking {num_omissions} omissions ({num_combinations} combinations)... ", end="", flush=True)
        
        found = False
        best_omitted = None
        
        for omitted_indices in itertools.combinations(range(total_inequalities), num_omissions):
            omitted_set = set(omitted_indices)
            if is_feasible_with_omissions(all_constraints, omitted_set, n):
                found = True
                best_omitted = omitted_set
                break
        
        if found:
            if verbose:
                print("FEASIBLE")
            ratio = num_omissions / total_inequalities
            if verbose:
                print(f"\nMinimum omissions needed: {num_omissions}")
                print(f"Total inequalities: {total_inequalities}")
                print(f"Ratio (min_omissions / total): {ratio:.6f}")
                print(f"Omitted constraint indices: {sorted(best_omitted)}")
            return num_omissions, total_inequalities, ratio, best_omitted
        else:
            if verbose:
                print("INFEASIBLE")
    
    # Should never reach here
    return total_inequalities, total_inequalities, 1.0, set(range(total_inequalities))


def print_omitted_constraints(X: list, omitted_indices: set):
    """
    Print the constraints that were omitted to achieve feasibility.
    """
    all_constraints = get_all_constraints(X)
    print("\nOmitted constraints:")
    for idx in sorted(omitted_indices):
        constraint = all_constraints[idx]
        print(f"  [{idx}]: {constraint[0]} {constraint[1]} {constraint[2]}")


if __name__ == "__main__":
    X = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    
    print("="*60)
    print("TSI Exact Lower Bound Analysis")
    print("="*60)
    print(f"X = {X}")
    print()
    
    min_omissions, total_ineq, ratio, omitted = tsi_exact_lower_bound(X, verbose=True)
    
    if omitted:
        print_omitted_constraints(X, omitted)
