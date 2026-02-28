import argparse
import os
from typing import Callable, Optional
import itertools
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from ten_ten_rl.core.pieces import PIECE_LIBRARY
from ten_ten_rl.core.adversarial_game import AdversarialGame
from ten_ten_rl.core.board import Board
from ten_ten_rl.core.bag import PieceBag
from ten_ten_rl.core.game import Game
from ten_ten_rl.env.ten_ten_env import TenTenEnv

from ten_ten_rl.scripts.ppo_cleanrl import Agent as PPOAgent
from ten_ten_rl.scripts.DQN import QNet as DQNAgent

from ten_ten_rl.scripts.greedy_play import play_episode


def make_env(adversarial: bool, seed: int) -> TenTenEnv:
    board = Board()
    bag = PieceBag(PIECE_LIBRARY, seed=seed)
    if adversarial:
        game = AdversarialGame(board, bag)
    else:
        game = Game(board, bag)
    env = TenTenEnv(game, invalid_move_penalty=-0.1)
    env.reset(seed=seed)
    return env


def main():
    parser = argparse.ArgumentParser(description="Run TenTen baselines")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument(
        "--ppo-path",
        type=str,
        required=False,
        help="Path to PPO checkpoint (state_dict)",
    )
    parser.add_argument("--ppo-hidden", type=int, default=512)
    parser.add_argument(
        "--dqn-path",
        type=str,
        required=False,
        help="Path to DQN checkpoint (state_dict)",
    )
    parser.add_argument("--dqn-hidden", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for adversarial, policy in itertools.product([True, False], ["greedy", "random"]):
        start_time = time.time()
        scores = [play_episode(make_env(adversarial=adversarial, seed=i), policy, seed=i) for i in range(args.episodes)]
        end_time = time.time()
        np.save(f"{adversarial}_{policy}_scores.npy", scores)
        np.save(f"{adversarial}_{policy}_times.npy", end_time - start_time)
        print(f"{adversarial}_{policy}: {np.mean(scores):.2f} ± {np.std(scores):.2f} in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
