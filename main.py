from typing import List
import numpy as np
import itertools


# --- Convert grid to NumPy boolean array ---
def grid_to_bool_array(grid: List[List[int]]) -> np.ndarray:
    return np.array(grid, dtype=bool)


# --- Convert NumPy boolean array back to 2D int list ---
def bool_array_to_grid(arr: np.ndarray) -> List[List[int]]:
    return arr.astype(int).tolist()


# --- Build Lights Out adjacency matrix ---
def build_matrix(rows: int, cols: int) -> np.ndarray:
    """Return an (rows*cols, rows*cols) boolean adjacency matrix."""
    n = rows * cols
    mat = np.zeros((n, n), dtype=bool)

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            mat[idx, idx] = True  # itself
            if r > 0:
                mat[idx, (r - 1) * cols + c] = True
            if r < rows - 1:
                mat[idx, (r + 1) * cols + c] = True
            if c > 0:
                mat[idx, r * cols + (c - 1)] = True
            if c < cols - 1:
                mat[idx, r * cols + (c + 1)] = True
    return mat


# --- Gaussian elimination in Z2 using NumPy ---
def gaussian_elimination(matrix: np.ndarray, rhs: np.ndarray) -> List[np.ndarray]:
    """Solve matrix * x = rhs in Z2 (boolean) and return all solutions."""
    mat = matrix.copy()
    rhs = rhs.copy()
    n_rows, n_cols = mat.shape
    pivot_cols = []

    for col in range(n_cols):
        pivot_row = None
        for row in range(col, n_rows):
            if mat[row, col]:
                pivot_row = row
                break
        if pivot_row is None:
            continue
        # Swap rows
        mat[[col, pivot_row]] = mat[[pivot_row, col]]
        rhs[[col, pivot_row]] = rhs[[pivot_row, col]]
        pivot_cols.append(col)
        # Eliminate other rows
        for row in range(n_rows):
            if row != col and mat[row, col]:
                mat[row] ^= mat[col]
                rhs[row] ^= rhs[col]

    # Check inconsistency
    for row in range(n_rows):
        if not mat[row].any() and rhs[row]:
            return []

    # Build one solution (free vars = 0)
    solution = np.zeros(n_cols, dtype=bool)
    for i, col in enumerate(pivot_cols):
        if rhs[i]:
            solution[col] = True

    # Enumerate all solutions by flipping free vars
    free_cols = set(range(n_cols)) - set(pivot_cols)
    all_solutions = [solution]
    for free_col in free_cols:
        new_solutions = []
        for sol in all_solutions:
            sol0, sol1 = sol.copy(), sol.copy()
            sol1[free_col] = True
            new_solutions.extend([sol0, sol1])
        all_solutions = new_solutions

    return all_solutions


# --- General solver ---
def solve_lights_out(initial_grid: List[List[int]], final_grid: List[List[int]]) -> List[List[List[int]]]:
    rows, cols = len(initial_grid), len(initial_grid[0])
    if len(final_grid) != rows or len(final_grid[0]) != cols:
        raise ValueError("Initial and final grids must have the same dimensions.")

    initial_state = grid_to_bool_array(initial_grid).flatten()
    final_state = grid_to_bool_array(final_grid).flatten()
    rhs = initial_state ^ final_state  # difference in Z2

    matrix = build_matrix(rows, cols)
    solutions = gaussian_elimination(matrix, rhs)

    if not solutions:
        print("No solution exists.")
        return []

    # Convert flattened solutions back to 2D grids
    return [bool_array_to_grid(sol.reshape(rows, cols)) for sol in solutions]


# --- Example usage ---
if __name__ == "__main__":
    initial_grid = [
        [1, 0],
        [0, 0]
    ]
    final_grid = [
        [0, 0],
        [0, 0]
    ]

    solutions = solve_lights_out(initial_grid, final_grid)

    if solutions:
        print(f"{len(solutions)} solution(s) found:")
        for sol_grid in solutions:
            for row in sol_grid:
                print(row)
            print("---")
