import argparse
import math
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from ten_ten_rl.core.bag import PieceBag
from ten_ten_rl.core.board import Board
from ten_ten_rl.core.game import Game
from ten_ten_rl.core.pieces import PIECE_LIBRARY
from ten_ten_rl.env.ten_ten_env import TenTenEnv

# Optional: wandb only if tracking is enabled
try:
    import wandb  # type: ignore
except ImportError:
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser(description="CleanRL-style PPO for TenTen")
    parser.add_argument("--total-timesteps", type=int, default=50_000_000)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--learning-rate-end", type=float, default=3e-5)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=3)
    parser.add_argument("--track", action="store_false")
    parser.add_argument("--wandb-project", type=str, default="tenten-ppo")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.1)  # 0.05
    parser.add_argument("--ent-coef-end", type=float, default=0.01)
    parser.add_argument("--anneal-ent", action="store_false")
    parser.add_argument("--vf-coef", type=float, default=0.3)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument(
        "--lr-schedule",
        type=str,
        choices=["linear", "cosine", "none"],
        default="cosine",
    )
    parser.add_argument(
        "--ent-schedule",
        type=str,
        choices=["linear", "cosine", "none"],
        default="cosine",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_false")
    parser.add_argument("--save-path", type=str, default="ppo_tenten.pt")
    return parser.parse_args()


def make_env(seed, idx):
    def thunk():
        board = Board()
        bag = PieceBag(PIECE_LIBRARY, seed=seed + idx)
        game = Game(board, bag)
        env = TenTenEnv(game)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed + idx)
        return env

    return thunk


def flatten_obs(obs):
    if isinstance(obs, dict):
        board = torch.as_tensor(obs["board"], dtype=torch.float32)
        hand = torch.as_tensor(obs["hand_masks"], dtype=torch.float32)
        board_flat = board.view(board.shape[0], -1)
        hand_flat = hand.view(hand.shape[0], -1)
        return torch.cat([board_flat, hand_flat], dim=1)
    board = torch.as_tensor(obs, dtype=torch.float32)
    return board.view(board.shape[0], -1)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=1024):  # 512
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        # MLP for flattened board + hand masks (dict observation)
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, self.hidden_size)),
            nn.GELU(),
            layer_init(nn.Linear(self.hidden_size, self.hidden_size)),
            nn.GELU(),
            # layer_init(nn.Linear(self.hidden_size, self.hidden_size)),
            # nn.GELU(),
            # layer_init(nn.Linear(self.hidden_size, self.hidden_size)),
            # nn.GELU(),
        )
        self.actor = layer_init(nn.Linear(self.hidden_size, self.action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(self.hidden_size, 1))

    def get_value(self, x):
        return self.critic(self.net(x))

    def get_action_and_value(self, x, mask=None, action=None):
        hidden = self.net(x)
        logits = self.actor(hidden)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        logprob = probs.log_prob(action)
        entropy = probs.entropy()
        return action, logprob, entropy, self.critic(hidden)
    
def schedule(schedule_type, begin, end, progress):
    if schedule_type == "linear":
        return begin + (end - begin) * progress
    elif schedule_type == "cosine":
        return begin + 0.5 * (end - begin) * (1.0 + math.cos(math.pi * progress))
    else:
        return begin

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed, i) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)

    board_size = int(np.prod(envs.single_observation_space["board"].shape))
    hand_size = int(np.prod(envs.single_observation_space["hand_masks"].shape))
    obs_dim = board_size + hand_size
    action_dim = envs.single_action_space.n

    agent = Agent(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.total_timesteps // batch_size
    run_name = f"tenten_ppo_seed{args.seed}_{int(time.time())}"

    if args.track:
        if wandb is None:
            raise ImportError("Install wandb or disable --track")
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
        )

    obs_buf = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    masks_buf = torch.zeros(
        (args.num_steps, args.num_envs, action_dim), device=device, dtype=torch.bool
    )

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    for update in range(1, num_updates + 1):
        episode_lengths = []
        episode_returns = []
        progress = (update - 1.0) / max(num_updates - 1, 1)
        lrnow = schedule(args.lr_schedule, args.learning_rate_end, args.learning_rate, progress)
        optimizer.param_groups[0]["lr"] = lrnow
        ent_coef_now = schedule(args.ent_schedule, args.ent_coef_end, args.ent_coef, progress)

        for step in range(args.num_steps):
            obs_t = flatten_obs(next_obs).to(device)
            # gymnasium SyncVectorEnv exposes `call` (not `env_method`)
            mask_np = np.stack(envs.call("action_mask"))
            mask_t = torch.as_tensor(mask_np, device=device)

            with torch.no_grad():
                action, logprob, entropy, value = agent.get_action_and_value(
                    obs_t, mask=mask_t
                )

            obs_buf[step] = obs_t
            masks_buf[step] = mask_t
            actions[step] = action
            logprobs[step] = logprob
            values[step] = value.flatten()
            dones[step] = next_done

            next_obs, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy()
            )
            # keep done flags in float to avoid bool subtraction later
            next_done = torch.as_tensor(
                np.logical_or(terminated, truncated), device=device, dtype=torch.float32
            )
            rewards[step] = torch.as_tensor(reward, device=device)

            global_step += args.num_envs
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ret = info["episode"]["r"]
                        length = info["episode"]["l"]
                        episode_lengths.append(length)
                        episode_returns.append(ret)
                        # print(
                        #     f"global_step={global_step}, return={ret}, length={length}"
                        # )

        with torch.no_grad():
            next_value = agent.get_value(flatten_obs(next_obs).to(device)).reshape(
                1, -1
            )
            advantages = torch.zeros_like(rewards)
            lastgaelam = torch.zeros(args.num_envs, device=device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
                advantages[t] = lastgaelam
            returns = advantages + values

        b_obs = obs_buf.reshape((-1, obs_dim))
        b_actions = actions.reshape((-1,))
        b_logprobs = logprobs.reshape((-1,))
        b_advantages = advantages.reshape((-1,))
        b_returns = returns.reshape((-1,))
        b_values = values.reshape((-1,))
        b_masks = masks_buf.reshape((-1, action_dim))

        clipfracs = []
        for epoch in range(args.update_epochs):
            idxs = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = idxs[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds],
                    mask=b_masks[mb_inds],
                    action=b_actions.long()[mb_inds],
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                mb_adv = b_advantages[mb_inds]
                # if mb_adv.std() > 0:
                #     mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std
                #     () + 1e-8)
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef_now * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                if args.target_kl and approx_kl > args.target_kl:
                    break

        y_pred, y_true = (
            b_values.detach().cpu().numpy(),
            b_returns.detach().cpu().numpy(),
        )
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        sps = int(global_step / (time.time() - start_time))
        mean_ep_len = np.mean(episode_lengths) if episode_lengths else float("nan")
        mean_ep_ret = np.mean(episode_returns) if episode_returns else float("nan")
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"update={update}/{num_updates} "
            f"loss={loss.item():.3f} "
            f"pg={pg_loss.item():.3f} "
            f"vf={v_loss.item():.3f} "
            f"ent={entropy_loss.item():.3f} "
            f"kl={approx_kl.item():.4f} "
            f"ev={explained_var:.3f} "
            f"SPS={sps} "
            f"clipfrac={np.mean(clipfracs):.3f} "
            f"mean_ep_len={mean_ep_len:.2f} "
            f"mean_ep_ret={mean_ep_ret:.2f} "
            f"lr={current_lr:.6f}"
        )
        if args.track:
            wandb.log(
                {
                    "charts/global_step": global_step,
                    "losses/total": loss.item(),
                    "losses/pg": pg_loss.item(),
                    "losses/vf": v_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/kl": approx_kl.item(),
                    "losses/clipfrac": float(np.mean(clipfracs)),
                    "losses/explained_variance": explained_var,
                    "charts/SPS": sps,
                    "charts/mean_ep_len": mean_ep_len,
                    "charts/mean_ep_ret": mean_ep_ret,
                    "charts/lr": current_lr,
                    "charts/ent_coef": ent_coef_now,
                },
                step=global_step,
            )
    fin_time = time.time()
    mean_ep_ret = np.mean(episode_returns) if episode_returns else float("nan")
    mean_ep_len = np.mean(episode_lengths) if episode_lengths else float("nan")
    torch.save(
        agent.state_dict(),
        f"{mean_ep_ret:.2f}_{mean_ep_len:.2f}_{fin_time}_{args.save_path}",
    )
    envs.close()


if __name__ == "__main__":
    main()
