import numpy as np

from ten_ten_rl.core.board import Board
from ten_ten_rl.core.bag import PieceBag
from ten_ten_rl.core.pieces import PIECE_LIBRARY
from ten_ten_rl.core.game import Game
from ten_ten_rl.env.ten_ten_env import TenTenEnv


def play_random_episode(seed=None):
    board = Board()
    bag = PieceBag(PIECE_LIBRARY, seed=seed)
    game = Game(board, bag)
    env = TenTenEnv(game)

    env.reset(seed=seed)

    total_reward = 0
    done = False

    while not done:
        legal = np.where(env.action_mask())[0]
        if len(legal) == 0:
            break

        action = int(np.random.choice(legal))
        _, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        done = terminated or truncated

    return total_reward


if __name__ == "__main__":
    scores = [play_random_episode(seed=i) for i in range(50)]
    print("Random average score:", np.mean(scores))
