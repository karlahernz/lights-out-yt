from typing import List

# --- Grid conversion ---
def grid_to_int(grid: List[List[int]]) -> int:
    """Convert a 2D grid of 0/1 to a single integer (bit-packed)."""
    rows, cols = len(grid), len(grid[0])
    return sum((1 << (r * cols + c)) for r, row in enumerate(grid) for c, val in enumerate(row) if val)

def int_to_grid(x: int, rows: int, cols: int) -> List[List[int]]:
    """Convert bit-packed integer back to 2D grid."""
    return [[(x >> (r * cols + c)) & 1 for c in range(cols)] for r in range(rows)]

# --- Matrix construction ---
def build_matrix(rows: int, cols: int) -> List[int]:
    """Build the lights-out adjacency matrix as bit-packed integers."""
    def neighbors(r, c):
        for dr, dc in ((0,0), (-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr * cols + nc

    return [sum(1 << pos for pos in neighbors(r, c)) for r in range(rows) for c in range(cols)]

# --- Gaussian elimination in Z2 ---
def gaussian_elimination(matrix: List[int], rhs: List[int], num_cells: int) -> List[int]:
    """Solve matrix * x = rhs in Z2. Returns list of bit-packed solutions."""
    n = len(matrix)
    m = num_cells
    mat = matrix[:]
    rhs = rhs[:]

    pivot_cols = []

    for col in range(m):
        # Find pivot row
        pivot_row = next((row for row in range(col, n) if (mat[row] >> col) & 1), None)
        if pivot_row is None:
            continue
        # Swap rows
        mat[col], mat[pivot_row] = mat[pivot_row], mat[col]
        rhs[col], rhs[pivot_row] = rhs[pivot_row], rhs[col]
        pivot_cols.append(col)
        # Eliminate other rows
        for row in range(n):
            if row != col and ((mat[row] >> col) & 1):
                mat[row] ^= mat[col]
                rhs[row] ^= rhs[col]

    # Check for inconsistency
    for row in range(n):
        if mat[row] == 0 and rhs[row]:
            return []

    # Build one solution (free vars = 0)
    solution = sum((1 << col) for i, col in enumerate(pivot_cols) if rhs[i])

    # Enumerate all solutions by flipping free variables
    free_cols = set(range(m)) - set(pivot_cols)
    all_solutions = [solution]
    for free_col in free_cols:
        new_solutions = []
        for sol in all_solutions:
            new_solutions.append(sol)               # free_col = 0
            new_solutions.append(sol | (1 << free_col))  # free_col = 1
        all_solutions = new_solutions

    return all_solutions

# --- General solver ---
def solve_lights_out(initial_grid: List[List[int]], final_grid: List[List[int]]) -> List[List[List[int]]]:
    rows, cols = len(initial_grid), len(initial_grid[0])
    if len(final_grid) != rows or len(final_grid[0]) != cols:
        raise ValueError("Final grid must have the same dimensions as initial grid")

    # Convert grids to integers (bit-packed)
    initial_state = grid_to_int(initial_grid)
    final_state   = grid_to_int(final_grid)
    diff = initial_state ^ final_state
    rhs = [int(b) for b in format(diff, f'0{rows*cols}b')[::-1]]  # LSB first

    matrix = build_matrix(rows, cols)
    solutions = gaussian_elimination(matrix, rhs, rows * cols)

    if not solutions:
        print("No solution exists.")
        return []

    # Convert integer solutions back to grids
    return [int_to_grid(sol, rows, cols) for sol in solutions]

# --- Example usage ---
if __name__ == "__main__":
    initial_grid = [
        [1,0],
        [0,0]
    ]
    final_grid = [
        [0,0],
        [0,0]
    ]

    solutions = solve_lights_out(initial_grid, final_grid)

    if solutions:
        print(f"{len(solutions)} solution(s) found:")
        for sol_grid in solutions:
            for row in sol_grid:
                print(row)
            print("---")
