"""
CSC 742 Evolutionary Computing
Assignment 1 (file 1): Hill Climbing on 8-Puzzle
Author: Matthew D. Branson
Date: 2025-01-16

Project Description:
This implementation demonstrates the Hill Climbing algorithm applied to the classic 8-puzzle problem.
"""
import random
import copy
import matplotlib.pyplot as plt
import numpy as np


# --- 8-Puzzle Visualization ---
def visualize_8_puzzle(board, cost):
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(0, 4, 1))
    ax.set_yticks(np.arange(0, 4, 1))
    ax.grid(True, color='black', linewidth=0.7)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for i in range(3):
        for j in range(3):
            if board[i][j] != 0:
                ax.text(j + 0.5, i + 0.5, str(board[i][j]), fontsize=20, ha='center', va='center')

    plt.title("8-Puzzle Configuration Cost: "+str(cost))
    plt.gca().invert_yaxis()
    plt.show()


# --- 8-Puzzle Algorithm ---
goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]


def generate_8_puzzle_instance(moves=20):
    def move_empty_tile(state):
        row, col = next((r, c) for r in range(3) for c in range(3) if state[r][c] == 0)
        possible_moves = [(row + dr, col + dc) for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]]
        valid_moves = [(r, c) for r, c in possible_moves if 0 <= r < 3 and 0 <= c < 3]
        new_row, new_col = random.choice(valid_moves)
        state[row][col], state[new_row][new_col] = state[new_row][new_col], state[row][col]
        return state

    state = copy.deepcopy(goal_state)
    for _ in range(moves):
        state = move_empty_tile(state)
    return state


def puzzle_heuristic(state):
    """
    Calculate the heuristic value for a given puzzle state.
    
    Uses the misplaced tiles heuristic: counts the number of tiles that are not
    in their goal position, excluding the blank tile (0).
    
    Args:
        state (list[list[int]]): Current puzzle state
        
    Returns:
        int: Number of misplaced tiles (0 indicates goal state reached)
    """
    misplaced = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0 and state[i][j] != goal_state[i][j]:
                misplaced += 1
    return misplaced


def generate_neighbors(state):
    """
    Generate all valid neighbor states from the current configuration.
    
    A neighbor is created by moving the blank tile in one of the four cardinal
    directions (up, down, left, right). Only valid moves within the 3x3 grid
    boundaries are included.
    
    Args:
        state (list[list[int]]): Current puzzle state
        
    Returns:
        list[list[list[int]]]: List of all valid neighbor states
    """
    neighbors = []
    # Find blank tile
    blank_row, blank_col = None, None
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                blank_row, blank_col = i, j
                break
    
    # Try moving blank in each direction
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    
    for dr, dc in moves:
        new_row = blank_row + dr
        new_col = blank_col + dc
        
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_state = copy.deepcopy(state)
            new_state[blank_row][blank_col], new_state[new_row][new_col] = \
                new_state[new_row][new_col], new_state[blank_row][blank_col]
            neighbors.append(new_state)
    
    return neighbors


def hill_climbing_puzzle(state):
    """
    Solve the 8-puzzle using the Hill Climbing algorithm.
    
    Implements steepest-ascent hill climbing by evaluating all neighbors and
    selecting the one with the lowest heuristic cost. The algorithm continues
    until reaching the goal state or getting stuck in a local minimum.
    
    Args:
        state (list[list[int]]): Initial puzzle state
        
    Returns:
        tuple: (final_state, success_flag)
            - final_state (list[list[int]]): Final puzzle configuration
            - success_flag (bool): True if goal reached, False if stuck in local minimum
    """
    current_state = copy.deepcopy(state)
    current_cost = puzzle_heuristic(current_state)
    
    print(f"Starting Hill Climbing with cost: {current_cost}")
    visualize_8_puzzle(current_state, current_cost)
    
    iteration = 0
    while current_cost > 0:
        neighbors = generate_neighbors(current_state)
        
        # Find best neighbor
        best_neighbor = None
        best_cost = current_cost
        
        for neighbor in neighbors:
            neighbor_cost = puzzle_heuristic(neighbor)
            if neighbor_cost < best_cost:
                best_neighbor = neighbor
                best_cost = neighbor_cost
        
        # If no improvement, we're stuck
        if best_neighbor is None:
            print(f"Stuck at local minimum after {iteration} iterations with cost {current_cost}")
            visualize_8_puzzle(current_state, current_cost)
            return current_state, False
        
        # Move to best neighbor
        current_state = best_neighbor
        current_cost = best_cost
        iteration += 1
        
        # Visualize progress
        print(f"Iteration {iteration}: Cost = {current_cost}")
        visualize_8_puzzle(current_state, current_cost)
    
    print(f"Goal reached after {iteration} iterations!")
    return current_state, True


# --- Running and Visualizing 8-Puzzle ---
initial_puzzle = generate_8_puzzle_instance()
visualize_8_puzzle(initial_puzzle, 0)
print(f"Initial 8-Puzzle: {initial_puzzle}")

# Q2: Run with at least 3 different initial states
if __name__ == "__main__":
    print("\n=== Q2.1: Testing with 3 different initial states ===")
    
    # Test 1
    print("\nTest 1:")
    puzzle1 = generate_8_puzzle_instance(moves=10)
    result1, success1 = hill_climbing_puzzle(puzzle1)
    
    # Test 2
    print("\nTest 2:")
    puzzle2 = generate_8_puzzle_instance(moves=20)
    result2, success2 = hill_climbing_puzzle(puzzle2)
    
    # Test 3
    print("\nTest 3:")
    puzzle3 = generate_8_puzzle_instance(moves=30)
    result3, success3 = hill_climbing_puzzle(puzzle3)
    
    # Q2.2: Demonstrate local minimum
    print("\n=== Q2.2: Demonstrating Local Minimum ===")
    # A configuration likely to get stuck - maybe? possibly? I hope?
    stuck_config = [[1, 2, 3], [4, 0, 6], [7, 5, 8]]
    print("Testing with a problematic configuration:")
    result_stuck, success_stuck = hill_climbing_puzzle(stuck_config)