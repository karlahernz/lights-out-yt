from cpu_based import solve_w_cpu

if __name__ == "__main__":
    initial_grid = [
        [1,0],
        [0,0]
    ]
    final_grid = [
        [0,0],
        [0,0]
    ]

    # --- cpu_based approach ---
    solutions = solve_w_cpu(initial_grid, final_grid)

    # if solutions:
    #     print(f"{len(solutions)} solution(s) found:")
    #     for sol_grid in solutions:
    #         for row in sol_grid:
    #             print(row)
    #         print("---")
