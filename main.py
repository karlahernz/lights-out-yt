def grid_to_int(grid):
    """Convert a 2D grid of 0/1 to a single integer (bit-packed)."""
    rows = len(grid)
    cols = len(grid[0])
    result = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c]:
                pos = r * cols + c
                result |= 1 << pos
    return result

def int_to_grid(x, rows, cols):
    """Convert integer back to 2D grid of size rows x cols."""
    grid = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            pos = r * cols + c
            if x & (1 << pos):
                grid[r][c] = 1
    return grid

def build_matrix(rows, cols):
    """Build the lights-out adjacency matrix (bit-packed rows)."""
    matrix = []
    for r in range(rows):
        for c in range(cols):
            row_mask = 0
            # toggle itself
            row_mask |= 1 << (r * cols + c)
            # toggle neighbors
            if r > 0:
                row_mask |= 1 << ((r-1) * cols + c)
            if r < rows - 1:
                row_mask |= 1 << ((r+1) * cols + c)
            if c > 0:
                row_mask |= 1 << (r * cols + (c-1))
            if c < cols - 1:
                row_mask |= 1 << (r * cols + (c+1))
            matrix.append(row_mask)
    return matrix

def gaussian_elimination(matrix, rhs, num_cells):
    """Solve matrix * x = rhs in Z2 using bit-packed rows."""
    n = len(matrix)
    m = num_cells
    mat = matrix[:]
    rhs = rhs[:]

    pivot_cols = []

    for col in range(m):
        # Find pivot row
        pivot_row = None
        for row in range(col, n):
            if (mat[row] >> col) & 1:
                pivot_row = row
                break
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

    # Check inconsistency
    for row in range(n):
        if mat[row] == 0 and rhs[row]:
            return []

    # Build one solution (free vars = 0)
    solution = 0
    for i, col in enumerate(pivot_cols):
        if rhs[i]:
            solution |= 1 << col

    # Enumerate all solutions by flipping free vars
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
def solve_lights_out(initial_grid, final_grid):
    rows = len(initial_grid)
    cols = len(initial_grid[0])
    if len(final_grid) != rows or len(final_grid[0]) != cols:
        raise ValueError("Final grid must have the same dimensions as initial grid")

    initial_state = grid_to_int(initial_grid)
    final_state   = grid_to_int(final_grid)
    rhs = [(initial_state ^ final_state) >> i & 1 for i in range(rows * cols)]

    matrix = build_matrix(rows, cols)
    solutions = gaussian_elimination(matrix, rhs, rows * cols)

    if not solutions:
        print("No solution exists.")
        return []

    # Convert integer solutions back to grid
    grid_solutions = [int_to_grid(sol, rows, cols) for sol in solutions]
    return grid_solutions

# --- Example usage ---
initial_grid = [
    [1,0,0],
    [0,1,0]
]

final_grid = [
    [0,0,0],
    [0,0,0]
]

solutions = solve_lights_out(initial_grid, final_grid)

if solutions:
    print(f"{len(solutions)} solution(s) found:")
    for sol_grid in solutions:
        for row in sol_grid:
            print(row)
        print("---")
