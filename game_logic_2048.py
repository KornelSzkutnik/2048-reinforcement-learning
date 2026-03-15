# 2048 game engine in log2 representation.

import pickle
import random

import numpy as np


def create_table():
    """
    Create a lookup table for every possible row (4 cells, each 0..15) and
    store the result of "move left": new row, score gained and whether anything changed.
    """
    table = {}
    # All combinations of 4 numbers in the range 0..15
    for a in range(16):
        for b in range(16):
            for c in range(16):
                for d in range(16):
                    score = 0
                    line = (a, b, c, d)
                    # If the row is empty or all 4 are different and non-zero, no merge is possible
                    if (len(set(line)) == 4 and min(line)) or (not max(line)):
                        table[line] = (line, score, False)
                        continue
                    # Copy only non-zero values
                    line_1 = [v for v in line if v]
                    # Merge equal adjacent values (two same next to each other -> one bigger, second becomes 0)
                    for i in range(len(line_1) - 1):
                        x = line_1[i]
                        if x == line_1[i + 1]:
                            score += 1 << (x + 1)  # points gained for merge
                            line_1[i] = x + 1
                            line_1[i + 1] = 0
                    # Keep only non-zero values and pad with zeros on the right
                    line_2 = [v for v in line_1 if v]
                    line_2 = tuple(line_2 + [0] * (4 - len(line_2)))
                    table[line] = (line_2, score, line != line_2)
    return table


class Game:
    """
    Single 2048 game in log2 representation.

    - row: 4x4 board, each cell 0 (empty) or 1,2,3... (tile 2,4,8,...)
    - score: total score gained from merges
    - odometer: number of moves made
    """

    # Shared "move row left" lookup table for all games
    table = create_table()

    def __init__(self, score=0, row=None):
        self.score = score
        self.odometer = 0
        self.moves = []
        self.tiles = []
        if row is None:
            # New game – empty board and two initial tiles
            self.row = np.zeros((4, 4), dtype=np.int32)
            self.new_tile()
            self.new_tile()
        else:
            self.row = np.array(row, dtype=np.int32).copy()
        self.starting_position = self.row.copy()

    def copy(self):
        return Game(self.score, self.row)

    @staticmethod
    def empty(row):
        """Return a list of (i, j) coordinates of empty cells."""
        zeros = np.where(row == 0)
        return list(zip(zeros[0], zeros[1]))

    @staticmethod
    def empty_count(row):
        """Return the number of empty cells (0–16)."""
        return 16 - np.count_nonzero(row)

    @staticmethod
    def adjacent_pair_count(row):
        """Return how many adjacent pairs have the same value (mergeable). 24 = all horizontal + vertical pairs."""
        diff_hor = row[:, :3] - row[:, 1:]
        diff_vert = row[:3, :] - row[1:, :]
        return 24 - np.count_nonzero(diff_hor) - np.count_nonzero(diff_vert)

    def game_over(self, row=None):
        """Game is over when there are no empty cells and no mergeable adjacent pairs."""
        r = row if row is not None else self.row
        if self.empty_count(r) > 0:
            return False
        if self.adjacent_pair_count(r) > 0:
            return False
        return True

    def create_new_tile(self, row):
        """Sample where to insert a new tile (2 with 90% prob., 4 with 10%) and return (value, (i, j))."""
        em = self.empty(row)
        tile = 1 if random.randrange(10) else 2  # 1 = tile 2, 2 = tile 4
        position = random.choice(em)
        return tile, position

    def new_tile(self):
        """Insert one random new tile on the board."""
        tile, position = self.create_new_tile(self.row)
        self.row[position] = tile
        self.tiles.append((tile, position))

    @staticmethod
    def _left(row, score):
        """
        Simulate "move all rows left" using the lookup table.
        Return (new_board, new_score, has_changed).
        """
        change = False
        new_row = row.copy()
        new_score = score
        for i in range(4):
            line = tuple(row[i])
            line_after, score_inc, change_line = Game.table[line]
            if change_line:
                change = True
                new_score += score_inc
                new_row[i] = line_after
        return new_row, new_score, change

    def pre_move(self, row, score, direction):
        """
        Simulate a single move in the given direction without mutating the game state.

        direction: 0=left, 1=up, 2=right, 3=down.
        Return (new_board, new_score, is_legal_move).

        The lookup table handles "move left"; other directions are implemented by
        rotating the board before and after the left move.
        """
        if direction == 0:
            new_row = row.copy()
        else:
            new_row = np.rot90(row, direction)
        new_row, new_score, change = self._left(new_row, score)
        if direction != 0:
            new_row = np.rot90(new_row, 4 - direction)
        return new_row, new_score, change

    def make_move(self, direction):
        """Apply a move to the real game (updates self.row and self.score)."""
        self.row, self.score, change = self.pre_move(self.row, self.score, direction)
        self.odometer += 1
        self.moves.append(direction)
        return change

    def save_game(self, file="saved_game.pkl"):
        with open(file, "wb") as f:
            pickle.dump(self, f, -1)

    @staticmethod
    def load_game(file="saved_game.pkl"):
        with open(file, "rb") as f:
            return pickle.load(f)

    def __str__(self):
        # Print board with values 2,4,8,... and below it: score, number of moves and max tile.
        lines = []
        for j in range(4):
            line_str = ""
            for val in self.row[j]:
                if val == 0:
                    line_str += "0\t"
                else:
                    line_str += str(1 << val) + "\t"
            lines.append(line_str)
        return (
            "\n".join(lines)
            + f"\n score = {self.score} moves = {self.odometer} reached {1 << np.max(self.row)}"
        )


def raw_board_to_log2(board_raw):
    """
    Convert a board in raw 2048 values (0, 2, 4, 8, ...) to log2 representation (0, 0, 1, 2, 3, ...).
    Needed when the agent plays using the `Board` class – the state must be passed in log2 format.
    """
    b = np.array(board_raw, dtype=np.int32)
    out = np.zeros_like(b)
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            if b[i, j] > 0:
                out[i, j] = int(np.log2(b[i, j]))
    return out
