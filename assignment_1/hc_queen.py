import random
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# --- 8-Queens Problem Visualization ---
def visualize_queens(board, cost):
    chessboard = np.zeros((8, 8))
    for col, row in enumerate(board):
        chessboard[row][col] = 1

    plt.figure(figsize=(6, 6))
    sns.heatmap(chessboard, cbar=False, annot=True, square=True, linewidths=0.5, linecolor='black', cmap='Blues',
                xticklabels=False, yticklabels=False)
    plt.title("8-Queens Solution Cost: "+str(cost))
    plt.show()


# --- 8-Queens Algorithm ---
def generate_8_queens_instance():
    return [random.randint(0, 7) for _ in range(8)]


def queens_heuristic(board):
    attacking_pairs = 0
    # Check all pairs of queens
    for i in range(8):
        for j in range(i + 1, 8):
            # Check if on same row
            if board[i] == board[j]:
                attacking_pairs += 1
            # Check if on same diagonal
            if abs(board[i] - board[j]) == abs(i - j):
                attacking_pairs += 1
    return attacking_pairs


def generate_neighbors(board):
    """Generate all possible neighbors by moving one queen at a time."""
    neighbors = []
    for col in range(8):
        current_row = board[col]
        for new_row in range(8):
            if new_row != current_row:
                new_board = board.copy()
                new_board[col] = new_row
                neighbors.append(new_board)
    return neighbors


def hill_climbing_queens(board):
    """Hill Climbing algorithm to solve 8-Queens problem."""
    current_board = board.copy()
    current_cost = queens_heuristic(current_board)
    
    print(f"Starting Hill Climbing with cost: {current_cost}")
    visualize_queens(current_board, current_cost)
    
    iteration = 0
    while current_cost > 0:
        neighbors = generate_neighbors(current_board)
        
        # Find best neighbor
        best_neighbor = None
        best_cost = current_cost
        
        for neighbor in neighbors:
            neighbor_cost = queens_heuristic(neighbor)
            if neighbor_cost < best_cost:
                best_neighbor = neighbor
                best_cost = neighbor_cost
        
        # If no improvement, we're stuck
        if best_neighbor is None:
            print(f"Stuck at local minimum after {iteration} iterations with cost {current_cost}")
            visualize_queens(current_board, current_cost)
            return current_board, False
        
        # Move to best neighbor
        current_board = best_neighbor
        current_cost = best_cost
        iteration += 1
        
        # Visualize progress
        print(f"Iteration {iteration}: Cost = {current_cost}")
        visualize_queens(current_board, current_cost)
    
    print(f"Solution found after {iteration} iterations!")
    return current_board, True


# --- Running and Visualizing 8-Queens ---
initial_queens = generate_8_queens_instance()
print(f"Initial Queens Board: {initial_queens}")
visualize_queens(initial_queens, 0)

# Q2: Run with at least 5 different initial states
if __name__ == "__main__":
    print("\n=== Q2: Evaluation with 5 different initial states ===")
    
    for run in range(5):
        print(f"\nRun {run + 1}:")
        initial_board = generate_8_queens_instance()
        initial_cost = queens_heuristic(initial_board)
        
        print(f"Initial state: {initial_board}")
        print(f"Initial heuristic value: {initial_cost}")
        
        final_board, success = hill_climbing_queens(initial_board)
        final_cost = queens_heuristic(final_board)
        
        print(f"Final state: {final_board}")
        print(f"Final heuristic value: {final_cost}")
        print(f"Success: {'Yes' if success else 'No (Local Minimum)'}")