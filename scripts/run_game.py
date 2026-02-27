import argparse
import os
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from ten_ten_rl.core.board import Board
from ten_ten_rl.core.bag import PieceBag
from ten_ten_rl.core.game import Game
from ten_ten_rl.env.ten_ten_env import TenTenEnv

from ten_ten_rl.scripts.ppo_cleanrl import Agent as PPOAgent
from ten_ten_rl.scripts.DQN import QNet as DQNAgent


def main():
    parser = argparse.ArgumentParser(description="Run TenTen baselines")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument(
        "--ppo-path",
        type=str,
        required=True,
        help="Path to PPO checkpoint (state_dict)",
    )
    parser.add_argument("--ppo-hidden", type=int, default=1024)
    parser.add_argument(
        "--dqn-path",
        type=str,
        required=True,
        help="Path to DQN checkpoint (state_dict)",
    )
    # parser.add_argument("--dqn-hidden", type=int, default=512)
    parser.add_argument("--device", type=str, default="gpu")
    args = parser.parse_args()

    device = torch.device(args.device)


if __name__ == "__main__":
    main()
