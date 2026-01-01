import numpy as np
from typing import List
import random


# -------------------------------
# Core solver functions
# -------------------------------

def grid_to_array(grid: List[List[int]]) -> np.ndarray:
    """Convert 2D grid to NumPy boolean array."""
    return np.array(grid, dtype=bool)


def array_to_grid(arr: np.ndarray) -> List[List[int]]:
    """Convert NumPy boolean array back to List[List[int]]."""
    return arr.astype(int).tolist()


def build_toggle_matrix(rows: int, cols: int) -> np.ndarray:
    """Build the Lights Out toggle matrix as a 2D boolean array."""
    n = rows * cols
    matrix = np.zeros((n, n), dtype=bool)

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            # toggle self
            matrix[idx, idx] = True
            # toggle neighbors
            if r > 0:  matrix[idx, (r - 1) * cols + c] = True
            if r < rows - 1: matrix[idx, (r + 1) * cols + c] = True
            if c > 0:  matrix[idx, r * cols + (c - 1)] = True
            if c < cols - 1: matrix[idx, r * cols + (c + 1)] = True
    return matrix


def gaussian_elimination(matrix: np.ndarray, rhs: np.ndarray) -> List[np.ndarray]:
    """Solve matrix * x = rhs in Z2 using boolean NumPy arrays."""
    mat = matrix.copy()
    b = rhs.copy()
    n_rows, n_cols = mat.shape
    pivot_cols = []
    row = 0

    # Forward elimination
    for col in range(n_cols):
        pivot_row = None
        for r in range(row, n_rows):
            if mat[r, col]:
                pivot_row = r
                break
        if pivot_row is None:
            continue  # free variable
        # Swap pivot row into place
        mat[[row, pivot_row]] = mat[[pivot_row, row]]
        b[[row, pivot_row]] = b[[pivot_row, row]]
        pivot_cols.append(col)
        # Eliminate all other rows
        for r in range(n_rows):
            if r != row and mat[r, col]:
                mat[r] ^= mat[row]
                b[r] ^= b[row]
        row += 1

    # Check inconsistency
    for r in range(row, n_rows):
        if not mat[r].any() and b[r]:
            return []  # no solution

    # Back-substitution
    solution = np.zeros(n_cols, dtype=bool)
    for i in reversed(range(len(pivot_cols))):
        col = pivot_cols[i]
        rhs_val = b[i]
        for j in range(i + 1, len(pivot_cols)):
            next_col = pivot_cols[j]
            if mat[i, next_col]:
                rhs_val ^= solution[next_col]
        solution[col] = rhs_val

    # Enumerate all solutions by free variables
    free_cols = set(range(n_cols)) - set(pivot_cols)
    all_solutions = [solution.copy()]
    for free_col in free_cols:
        new_solutions = []
        for sol in all_solutions:
            new_solutions.append(sol.copy())  # free_col = 0
            new_sol = sol.copy()
            new_sol[free_col] = True
            new_solutions.append(new_sol)
        all_solutions = new_solutions

    return all_solutions


def solve_lights_out(initial_grid: List[List[int]], final_grid: List[List[int]]) -> List[List[List[int]]]:
    rows, cols = len(initial_grid), len(initial_grid[0])
    init_arr = grid_to_array(initial_grid)
    final_arr = grid_to_array(final_grid)

    n_cells = rows * cols
    rhs = (init_arr ^ final_arr).reshape(n_cells)

    matrix = build_toggle_matrix(rows, cols)
    solutions = gaussian_elimination(matrix, rhs)
    return [array_to_grid(sol.reshape(rows, cols)) for sol in solutions]


# -------------------------------
# Example usage
# -------------------------------

def random_grid(rows: int, cols: int) -> List[List[int]]:
    return [[random.choice((0, 1)) for _ in range(cols)] for _ in range(rows)]


def print_grid(grid: List[List[int]]) -> None:
    for row in grid:
        print(" ".join(str(cell) for cell in row))


if __name__ == "__main__":
    rows, cols = 5, 5  # small example
    # initial_grid = random_grid(rows, cols)
    initial_grid = [[1 for _ in range(cols)] for _ in range(rows)]
    final_grid = [[0] * cols for _ in range(rows)]  # goal: all off

    print("Initial Grid:")
    print_grid(initial_grid)
    print("\nSolving...")

    solutions = solve_lights_out(initial_grid, final_grid)

    if not solutions:
        print("\nNo solution exists.")
    else:
        print(f"\n{len(solutions)} solution(s) found:")
        for i, sol in enumerate(solutions, 1):
            print(f"\nSolution {i}:")
            print_grid(sol)
