# Temporal Difference (TD(0)) agent implementation for 2048.

import pickle
import time
from collections import deque

import numpy as np

from game_logic_2048 import Game


def f_2(x):
    """Feature function based on vertical and horizontal tile pairs."""
    features = []
    # Vertical pairs
    for i in range(3):
        for j in range(4):
            features.append(x[i, j] * 16 + x[i + 1, j])

    # Horizontal pairs
    for i in range(4):
        for j in range(3):
            features.append(x[i, j] * 16 + x[i, j + 1])

    return np.array(features)


FEATURE_FUNCTIONS = {2: f_2}
PARAMETER_SHAPE = {2: (24, 16**2)}


class TDAgent:
    """TD(0) agent learning a linear value function V(s)."""

    def __init__(
        self,
        name="agent",
        alpha=0.25,
        decay=0.75,
        decay_step=10000,
        low_alpha_limit=0.01,
        with_weights=True,
    ):
        self.name = name
        self.file = name + ".pkl"
        self.game_file = "best_of_" + self.file
        self.n = 2
        self.alpha = alpha
        self.decay = decay
        self.decay_step = decay_step
        self.low_alpha_limit = low_alpha_limit
        self.num_feat, self.size_feat = PARAMETER_SHAPE[2]
        self.features = FEATURE_FUNCTIONS[2]
        self.step = 0
        self.top_game = None
        self.top_score = 0
        self.train_history = []
        self.next_decay = decay_step
        self.top_tile = 10

        if with_weights:
            self.init_weights()
        else:
            self.weights = None

    def __str__(self):
        return f"Agent {self.name}, n={self.n}\ntrained for {self.step} episodes, top score = {self.top_score}"

    def init_weights(self):
        self.weights = (np.random.random((self.num_feat, self.size_feat)) / 100).tolist()

    def list_to_np(self):
        return [np.array(w, dtype=np.float32) for w in self.weights]

    def np_to_list(self):
        self.weights = [w.tolist() for w in self.weights]

    def evaluate(self, row):
        """Compute V(s) as the sum of weights for active features."""
        return sum(self.weights[i][int(f)] for i, f in enumerate(self.features(row)))

    def update(self, row, dw):
        """TD(0) weight update with D4 symmetry (rotations and transpositions)."""
        for _ in range(4):
            for i, f in enumerate(self.features(row)):
                self.weights[i][int(f)] += dw
            row = np.transpose(row)
            for i, f in enumerate(self.features(row)):
                self.weights[i][int(f)] += dw
            row = np.rot90(np.transpose(row))

    def episode(self):
        """Single training episode: greedy move selection and TD(0) updates."""
        game = Game()
        state, old_label = None, 0

        while not game.game_over(game.row):
            action, best_value = 0, -np.inf
            best_row, best_score = None, None

            for direction in range(4):
                new_row, new_score, change = game.pre_move(game.row, game.score, direction)
                if change:
                    value = self.evaluate(new_row)
                    if value > best_value:
                        action, best_value = direction, value
                        best_row, best_score = new_row, new_score

            if state is not None:
                reward = best_score - game.score
                scaled_reward = np.log2(1 + reward)
                dw = (scaled_reward + best_value - old_label) * self.alpha / self.num_feat
                self.update(state, dw)

            game.row, game.score = best_row, best_score
            game.odometer += 1
            game.moves.append(action)
            state, old_label = game.row.copy(), best_value
            game.new_tile()

        game.moves.append(-1)
        PENALTY = -10
        dw = (PENALTY - old_label) * self.alpha / self.num_feat
        self.update(state, dw)
        self.step += 1
        return game

    def decay_alpha(self):
        self.alpha = round(max(self.alpha * self.decay, self.low_alpha_limit), 4)
        self.next_decay = self.step + self.decay_step
        print("------")
        print(f"episode = {self.step + 1}, current learning rate = {round(self.alpha, 4)}:")
        print("------")

    def save_agent(self):
        self.weights = self.list_to_np()
        with open(self.file, "wb") as file:
            pickle.dump(self, file, -1)
        self.np_to_list()

    def save_game(self, game):
        game.save_game(self.game_file)

    @staticmethod
    def load_agent(file):
        with open(file, "rb") as file:
            agent = pickle.load(file)
        agent.np_to_list()
        return agent

    def train_run(self, num_eps=50000, add_weights="already", saving=True):
        if add_weights == "add":
            self.init_weights()

        av1000, ma100 = [], deque(maxlen=100)
        reached = [0] * 7
        best_of_1000 = Game()
        global_start = start = time.time()

        print(f"Agent {self.name} training started, current step = {self.step}")

        for i in range(self.step + 1, self.step + num_eps + 2):
            if self.step > self.next_decay and self.alpha > self.low_alpha_limit:
                self.decay_alpha()

            game = self.episode()
            ma100.append(game.score)
            av1000.append(game.score)

            if game.score > best_of_1000.score:
                best_of_1000 = game
                if game.score > self.top_score:
                    self.top_game, self.top_score = game, game.score
                    print(f"\nnew best game at episode {i}!\n{game}\n")
                    if saving:
                        self.save_game(game)
                        print(f"game saved at {self.game_file}")

            max_tile = np.max(game.row)
            if max_tile >= 10:
                reached[max_tile - 10] += 1

            if max_tile > self.top_tile:
                self.top_tile = max_tile
                self.decay_alpha()

            if i % 100 == 0:
                ma = int(np.mean(ma100))
                self.train_history.append(ma)
                print(f"episode {i}: score {game.score} reached {1 << max_tile} ma_100 = {ma}")

            if i % 1000 == 0:
                average = np.mean(av1000)
                print("\n------")
                print(f"{round((time.time() - start) / 60, 2)} min")
                start = time.time()
                print(f"episode = {i}")
                print(f"average over last 1000 episodes = {average}")
                av1000 = []
                for j in range(7):
                    r = sum(reached[j:]) / 10
                    if r:
                        print(f"{1 << (j + 10)} reached in {r} %")
                reached = [0] * 7
                print(f"best of last 1000:\n{best_of_1000}")
                print(f"best of this Agent:\n{self.top_game}")
                print(f"learning rate = {round(self.alpha, 4)}")
                print("------\n")
                if saving:
                    self.save_agent()
                    print(f"agent saved in {self.file}")
                best_of_1000 = Game()

        total_time = int(time.time() - global_start)
        print(f"Total time = {total_time // 60} min {total_time % 60} sec")
        if saving:
            self.save_agent()
            print(f"{self.name} saved at step {self.step} in {self.file}\n")

    def get_best_action(self, row, score):
        """For a given board (log2) and score, return (direction, value)."""
        best_dir, best_value = 0, -np.inf
        for direction in range(4):
            new_row, new_score, change = Game().pre_move(row, score, direction)
            if change:
                value = self.evaluate(new_row)
                if value > best_value:
                    best_dir, best_value = direction, value
        return best_dir, best_value
