import random
import copy
import matplotlib.pyplot as plt
import seaborn as sns
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
    # compute misplaced items
    misplaced = 0
    return misplaced


def hill_climbing_puzzle(state):
    pass


# --- Running and Visualizing 8-Puzzle ---
initial_puzzle = generate_8_puzzle_instance()
visualize_8_puzzle(initial_puzzle, 0)
print(f"Initial 8-Puzzle: {initial_puzzle}")