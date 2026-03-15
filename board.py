from random import randint

import numpy as np


class Board:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        # Generate 2 tiles at the start of the game
        self.generate()
        self.generate()

    def reset(self):
        """Reset the board to the initial state and return the state."""
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.generate()
        self.generate()
        return self.get_state()

    def get_state(self):
        """Return the current board state as a NumPy array."""
        return self.board.copy()

    def generate(self):
        # Check if there are any empty cells
        if 0 not in self.board:
            return False

        ran_x = randint(0, 3)
        ran_y = randint(0, 3)
        while self.board[ran_x, ran_y] != 0:
            ran_x = randint(0, 3)
            ran_y = randint(0, 3)

        if randint(0, 100) <= 90:
            self.board[ran_x, ran_y] = 2  # 90% probability -> 2
        else:
            self.board[ran_x, ran_y] = 4  # 10% probability -> 4

        return True

    def is_game_over(self):
        if 0 in self.board:
            return False

        # Check if any adjacent tiles can be merged
        for i in range(4):
            for j in range(4):
                current = self.board[i][j]
                if j < 3 and current == self.board[i][j + 1]:
                    return False
                if i < 3 and current == self.board[i + 1][j]:
                    return False

        return True

    def has_won(self):
        return 2048 in self.board

    def get_max_tile(self):
        return np.max(self.board)

    def move_left(self):
        old_board = self.board.copy()
        old_score = self.score

        for i in range(0, 4):
            row = self.board[i, :]
            non_zero = row[row != 0]
            zeros_needed = len(row) - len(non_zero)
            non_zero = np.pad(non_zero, (0, zeros_needed), mode="constant", constant_values=0)

            k = 0
            while k < 3:
                if non_zero[k] == non_zero[k + 1] and non_zero[k] != 0:
                    non_zero[k] *= 2
                    self.score += non_zero[k]
                    non_zero[k + 1] = 0
                    k += 1
                k += 1

            non_zero = non_zero[non_zero != 0]
            zeros_needed = len(row) - len(non_zero)
            non_zero = np.pad(non_zero, (0, zeros_needed), mode="constant", constant_values=0)
            self.board[i, :] = non_zero

        moved = not np.array_equal(old_board, self.board)
        reward = self.score - old_score if moved else 0
        return moved, reward

    def move_right(self):
        old_board = self.board.copy()
        old_score = self.score

        for i in range(0, 4):
            row = self.board[i, :]
            non_zero = row[row != 0]
            non_zero = np.flip(non_zero, axis=0)
            zeros_needed = len(row) - len(non_zero)
            non_zero = np.pad(non_zero, (0, zeros_needed), mode="constant", constant_values=0)

            k = 0
            while k < 3:
                if non_zero[k] == non_zero[k + 1] and non_zero[k] != 0:
                    non_zero[k] *= 2
                    self.score += non_zero[k]
                    non_zero[k + 1] = 0
                    k += 1
                k += 1

            non_zero = np.flip(non_zero, axis=0)
            non_zero = non_zero[non_zero != 0]
            zeros_needed = len(row) - len(non_zero)
            non_zero = np.pad(non_zero, (zeros_needed, 0), mode="constant", constant_values=0)
            self.board[i, :] = non_zero

        moved = not np.array_equal(old_board, self.board)
        reward = self.score - old_score if moved else 0
        return moved, reward

    def move_down(self):
        old_board = self.board.copy()
        old_score = self.score

        self.board = np.transpose(self.board)

        for i in range(0, 4):
            row = self.board[i, :]
            non_zero = row[row != 0]
            zeros_needed = len(row) - len(non_zero)
            non_zero = np.pad(non_zero, (0, zeros_needed), mode="constant", constant_values=0)

            k = 0
            while k < 3:
                if non_zero[k] == non_zero[k + 1] and non_zero[k] != 0:
                    non_zero[k] *= 2
                    self.score += non_zero[k]
                    non_zero[k + 1] = 0
                    k += 1
                k += 1

            non_zero = non_zero[non_zero != 0]
            zeros_needed = len(row) - len(non_zero)
            non_zero = np.pad(non_zero, (zeros_needed, 0), mode="constant", constant_values=0)
            self.board[i, :] = non_zero
        self.board = np.transpose(self.board)

        moved = not np.array_equal(old_board, self.board)
        reward = self.score - old_score if moved else 0
        return moved, reward

    def move_up(self):
        old_board = self.board.copy()
        old_score = self.score

        self.board = np.transpose(self.board)

        for i in range(0, 4):
            row = self.board[i, :]
            non_zero = row[row != 0]
            zeros_needed = len(row) - len(non_zero)
            non_zero = np.pad(non_zero, (0, zeros_needed), mode="constant", constant_values=0)

            k = 0
            while k < 3:
                if non_zero[k] == non_zero[k + 1] and non_zero[k] != 0:
                    non_zero[k] *= 2
                    self.score += non_zero[k]
                    non_zero[k + 1] = 0
                    k += 1
                k += 1

            non_zero = non_zero[non_zero != 0]
            zeros_needed = len(row) - len(non_zero)
            non_zero = np.pad(non_zero, (0, zeros_needed), mode="constant", constant_values=0)
            self.board[i, :] = non_zero
        self.board = np.transpose(self.board)

        moved = not np.array_equal(old_board, self.board)
        reward = self.score - old_score if moved else 0
        return moved, reward
