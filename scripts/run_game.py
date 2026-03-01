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


def flatten_obs(obs: dict) -> torch.Tensor:
    board = torch.as_tensor(obs["board"], dtype=torch.float32)
    hand = torch.as_tensor(obs["hand_masks"], dtype=torch.float32)
    return torch.cat([board.reshape(-1), hand.reshape(-1)], dim=0)


def obs_to_torch(obs: dict, device: str):
    board = (
        torch.as_tensor(obs["board"], dtype=torch.float32, device=device)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    hand = torch.as_tensor(
        obs["hand_masks"], dtype=torch.float32, device=device
    ).unsqueeze(0)
    return board, hand


def evaluate_policy(
    episodes: int,
    policy_fn: Callable[[dict, np.ndarray], int],
    seed_offset: int = 0,
    adversarial: bool = False,
) -> float:
    scores = []
    times = []
    for ep in range(episodes):
        if ep % 100 == 0:
            print(f"Evaluating policy for {ep} episodes")
        start_time = time.time()
        env = make_env(adversarial=adversarial, seed=seed_offset + ep)
        obs, _ = env.reset(seed=seed_offset + ep)
        done = False
        total = 0.0
        steps = 0
        while not done:
            mask = env.action_mask().astype(bool)
            action = policy_fn(obs, mask)
            if action is None:
                break
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            steps += 1
            done = terminated or truncated
        scores.append(float(total))
        times.append(time.time() - start_time)
        env.close()
    return scores, np.sum(times)


def main():
    parser = argparse.ArgumentParser(description="Run TenTen baselines")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--ppo-hidden", type=int, default=1024)
    # parser.add_argument("--dqn-hidden", type=int, default=1024)
    parser.add_argument("--run-random", action="store_true")
    parser.add_argument("--run-greedy", action="store_true")
    parser.add_argument("--run-ppo", action="store_true")
    parser.add_argument("--run-dqn", action="store_true")
    parser.add_argument("--run-dqn-nopr", action="store_true")
    args = parser.parse_args()
    ppo_path = "ten_ten_rl/models/score502_1772003097_ppo_tenten.pt"
    dqn_path = "ten_ten_rl/models/dqn_model.pt"
    dqn_path_nopr = "ten_ten_rl/models/False_best_dqn_model_1038.00.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    basic_runs = []
    if args.run_random:
        basic_runs.append("random")
    if args.run_greedy:
        basic_runs.append("greedy")

    for adversarial, policy in itertools.product([True, False], basic_runs):
        start_time = time.time()
        scores = []
        for i in range(args.episodes):
            scores.append(
                play_episode(make_env(adversarial=adversarial, seed=i), policy, seed=i)
            )
        end_time = time.time()
        np.save(f"{adversarial}_{policy}_scores.npy", scores)
        np.save(f"{adversarial}_{policy}_elapsed.npy", end_time - start_time)
        print(
            f"{adversarial}_{policy}: {np.mean(scores):.2f} ± {np.std(scores):.2f} in {end_time - start_time:.2f} seconds"
        )

    env0 = make_env(adversarial=False, seed=0)
    obs0, _ = env0.reset(seed=0)
    action_dim = env0.action_space.n
    env0.close()

    if args.run_ppo:
        obs_dim = flatten_obs(obs0).numel()
        ppo_agent = PPOAgent(obs_dim, action_dim, hidden_size=args.ppo_hidden).to(
            device
        )
        ppo_agent.load_state_dict(torch.load(ppo_path, map_location=device))
        ppo_agent.eval()

        def ppo_greedy_policy(obs, mask):
            obs_t = flatten_obs(obs).to(device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            with torch.no_grad():
                hidden = ppo_agent.net(obs_t)
                logits = ppo_agent.actor(hidden)
                logits = logits.masked_fill(~mask_t, -1e9)
                action = torch.argmax(logits).item()
            return action

        for adversarial in [True, False]:
            scores, elapsed = evaluate_policy(
                args.episodes, ppo_greedy_policy, adversarial=adversarial
            )
            np.save(f"ppo_scores_{adversarial}.npy", scores)
            np.save(f"ppo_elapsed_{adversarial}.npy", elapsed)
            print(
                f"ppo: mean={np.mean(scores):.2f} std={np.std(scores):.2f} time={elapsed:.2f}s"
            )

    if args.run_dqn:
        dqn_agent = DQNAgent(n_actions=action_dim).to(device)
        dqn_agent.load_state_dict(torch.load(dqn_path, map_location=device))
        dqn_agent.eval()

        def dqn_greedy_policy(obs, mask):
            board_t, hand_t = obs_to_torch(obs, device)
            with torch.no_grad():
                q = dqn_agent(board_t, hand_t).squeeze(0)
                mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
                q[~mask_t] = -1e9
                return int(torch.argmax(q).item())

        for adversarial in [True, False]:
            scores, elapsed = evaluate_policy(
                args.episodes, dqn_greedy_policy, adversarial=adversarial
            )
            np.save(f"dqn_scores_{adversarial}.npy", scores)
            np.save(f"dqn_elapsed_{adversarial}.npy", elapsed)
            print(
                f"dqn: mean={np.mean(scores):.2f} std={np.std(scores):.2f} time={elapsed:.2f}s"
            )

    if args.run_dqn_nopr:
        dqn_agent = DQNAgent(n_actions=action_dim).to(device)
        dqn_agent.load_state_dict(torch.load(dqn_path_nopr, map_location=device))
        dqn_agent.eval()

        def dqn_greedy_policy(obs, mask):
            board_t, hand_t = obs_to_torch(obs, device)
            with torch.no_grad():
                q = dqn_agent(board_t, hand_t).squeeze(0)
                mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
                q[~mask_t] = -1e9
                return int(torch.argmax(q).item())

        for adversarial in [True, False]:
            if adversarial:
                scores, elapsed = evaluate_policy(
                    1, dqn_greedy_policy, adversarial=adversarial
                )
            else:
                scores, elapsed = evaluate_policy(
                    args.episodes, dqn_greedy_policy, adversarial=adversarial
                )
            np.save(f"nopr_dqn_scores_{adversarial}.npy", scores)
            np.save(f"nopr_dqn_elapsed_{adversarial}.npy", elapsed)
            print(
                f"nopr_dqn: mean={np.mean(scores):.2f} std={np.std(scores):.2f} time={elapsed:.2f}s"
            )


if __name__ == "__main__":
    main()
