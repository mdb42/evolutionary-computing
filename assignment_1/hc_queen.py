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
    # your code
    return attacking_pairs


def hill_climbing_queens(board):
    pass


# --- Running and Visualizing 8-Queens ---
initial_queens = generate_8_queens_instance()
print(f"Initial Queens Board: {initial_queens}")
visualize_queens(initial_queens, 0)
