import numpy as np

from ten_ten_rl.core.board import Board
from ten_ten_rl.core.bag import PieceBag
from ten_ten_rl.core.pieces import PIECE_LIBRARY
from ten_ten_rl.core.game import Game
from ten_ten_rl.env.ten_ten_env import TenTenEnv


def make_env(seed=0):
    board = Board() # here we can change board size with size=n 
    bag = PieceBag(PIECE_LIBRARY, seed=seed)
    game = Game(board, bag)
    return TenTenEnv(game) 


def greedy_action(env):
    """Pick the legal move with the best immediate score."""
    game = env.game
    board = game.board
    N = env.N

    best_action = None
    best_value = -1e9

    for pi, piece in enumerate(game.hand):
        for r in range(N - piece.height + 1):
            for c in range(N - piece.width + 1):
                if not board.can_place(piece, r, c):
                    continue

                b2 = board.copy()
                info = b2.place(piece, r, c)

                value = info["placed_blocks"] + 10 * (info["cleared_rows"] + info["cleared_cols"])

                action = pi * (N * N) + r * N + c  # matches new env encoding
                if value > best_value:
                    best_value = value
                    best_action = action

    return best_action


def play_episode(env, policy="random", seed=0, max_steps=5000):
    env.reset(seed=seed)

    total_reward = 0.0
    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated) and steps < max_steps:
        if policy == "random":
            legal = np.where(env.action_mask())[0]
            if len(legal) == 0:
                break
            action = int(np.random.choice(legal))

        elif policy == "greedy":
            action = greedy_action(env)
            if action is None:
                break

        else:
            raise ValueError("policy must be 'random' or 'greedy'")

        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1

    return total_reward


if __name__ == "__main__":
    env = make_env()

    n_eps = 200
    rand_scores = [play_episode(env, "random", seed=i) for i in range(n_eps)]
    greedy_scores = [play_episode(env, "greedy", seed=i) for i in range(n_eps)]

    print(f"Random avg: {np.mean(rand_scores):.2f} ± {np.std(rand_scores):.2f}")
    print(f"Greedy avg: {np.mean(greedy_scores):.2f} ± {np.std(greedy_scores):.2f}")

    # Watch one greedy game (prints ASCII each step)
    env_watch = make_env(seed=123)
    env_watch.reset(seed=123)

    done = False
    while not done:
        a = greedy_action(env_watch)
        if a is None:
            break
        _, r, terminated, truncated, _ = env_watch.step(a)
        env_watch.render()
        done = terminated or truncated
