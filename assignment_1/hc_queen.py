"""
CSC 742 Evolutionary Computing
Assignment 1 (file 2): Hill Climbing on 8-Queens
Author: Matthew D. Branson
Date: 2025-06-16

Project Description:
This implementation demonstrates the Hill Climbing algorithm applied to the classic 8-Queens problem.
"""
import random
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# --- 8-Queens Problem Visualization ---
# Adapted for selective figure saving
def visualize_queens(board, cost, save_as=None):
    chessboard = np.zeros((8, 8))
    for col, row in enumerate(board):
        chessboard[row][col] = 1

    plt.figure(figsize=(6, 6))
    sns.heatmap(chessboard, cbar=False, annot=True, square=True, linewidths=0.5, linecolor='black', cmap='Blues',
                xticklabels=False, yticklabels=False)
    plt.title("8-Queens Solution Cost: "+str(cost))
    
    # Added this because I got tired of sorting through the plots
    if save_as:
        plt.savefig(save_as)
        print(f"Saved image as: {save_as}")
    
    plt.show()


# --- 8-Queens Algorithm ---
def generate_8_queens_instance():
    return [random.randint(0, 7) for _ in range(8)]


def queens_heuristic(board):
    """
    Calculate the heuristic value (number of attacking pairs) for a board configuration.
    
    Examines all pairs of queens and counts how many pairs are attacking each other.
    Queens attack if they are on the same row or diagonal.
    
    Args:
        board (list[int]): Current board configuration
        
    Returns:
        int: Number of attacking queen pairs (0 indicates a valid solution)
    """
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
    """
    Generate all possible neighbor states by moving one queen at a time.
    
    For each queen, generates all possible states where that queen is moved to
    a different row within its column. This produces 8 * 7 = 56 total neighbors.
    
    Args:
        board (list[int]): Current board configuration
        
    Returns:
        list[list[int]]: List of all possible neighbor configurations
    """
    neighbors = []
    for col in range(8):
        current_row = board[col]
        for new_row in range(8):
            if new_row != current_row:
                new_board = board.copy()
                new_board[col] = new_row
                neighbors.append(new_board)
    return neighbors


def hill_climbing_queens(board, save_stuck=False):
    """
    Solve the 8-Queens problem using the Hill Climbing algorithm.
    
    Implements steepest-ascent hill climbing by evaluating all neighbors and
    selecting the one with the fewest attacking pairs. The algorithm continues
    until finding a solution (0 attacking pairs) or getting stuck in a local minimum.
    
    Args:
        board (list[int]): Initial board configuration
        save_stuck (bool): If True, saves visualization when stuck in local minimum
        
    Returns:
        tuple: (final_board, success_flag)
            - final_board (list[int]): Final board configuration
            - success_flag (bool): True if solution found, False if stuck in local minimum
    """
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
            if save_stuck:
                visualize_queens(current_board, current_cost, save_as="queens_stuck.png")
            else:
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
visualize_queens(initial_queens, queens_heuristic(initial_queens))

# Q2: Run with at least 5 different initial states
if __name__ == "__main__":
    print("\n=== Q2: Evaluation with 5 different initial states ===")
    
    stuck_saved = False  # Flag to save only the first stuck case.
    
    for run in range(5):
        print(f"\nRun {run + 1}:")
        initial_board = generate_8_queens_instance()
        initial_cost = queens_heuristic(initial_board)
        
        print(f"Initial state: {initial_board}")
        print(f"Initial heuristic value: {initial_cost}")
        
        final_board, success = hill_climbing_queens(initial_board, save_stuck=(not stuck_saved and run < 5))
        if not success and not stuck_saved:
            stuck_saved = True
        
        final_cost = queens_heuristic(final_board)
        
        print(f"Final state: {final_board}")
        print(f"Final heuristic value: {final_cost}")
        print(f"Success: {'Yes' if success else 'No (Local Minimum)'}")
    
    # Solutions do exist with random restarts
    print("\n=== Random Restarts ===")
    print("Trying multiple random starts to find a solution...")
    
    attempts = 0
    max_attempts = 100
    
    while attempts < max_attempts:
        attempts += 1
        initial = generate_8_queens_instance()
        final, success = hill_climbing_queens(initial, save_stuck=False)
        
        if success:
            print(f"\nFound solution after {attempts} random restarts!")
            print(f"Solution: {final}")
            print(f"Cost: {queens_heuristic(final)}")
            visualize_queens(final, 0, save_as="queens_success.png")
            break
        
        if attempts % 10 == 0:
            print(f"Tried {attempts} restarts...")
    
    if attempts == max_attempts:
        print(f"\nNo solution found in {max_attempts} attempts")