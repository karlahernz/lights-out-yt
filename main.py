import random
from typing import List
from cpu_based import solve_w_cpu

def random_grid(rows: int, cols: int) -> List[List[int]]:
    return [[random.choice((0, 1)) for _ in range(cols)] for _ in range(rows)]

def print_grid(grid: List[List[int]]) -> None:
    for row in grid:
        print(row)

if __name__ == "__main__":
    initial_grid = random_grid(50, 50)
    final_grid = random_grid(50, 50)

    # --- cpu_based approach ---
    solutions = solve_w_cpu(initial_grid, final_grid)

    # Return if no solutions
    if not solutions:
        print("\nNo solution exists.")

    if solutions:
        print(f"\n{len(solutions)} solution(s) found:")
        print("---")
        for sol_grid in solutions:
            print_grid(sol_grid)
            print("---")



