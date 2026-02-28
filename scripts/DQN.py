import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torch.optim as optim
import time
import pygame
import wandb
t0 = time.time()
last_t = t0
last_step = 0

LOG_EVERY_STEPS = 5000



from ten_ten_rl.core.board import Board
from ten_ten_rl.core.bag import PieceBag
from ten_ten_rl.core.pieces import PIECE_LIBRARY
from ten_ten_rl.core.game import Game
from ten_ten_rl.env.ten_ten_env import TenTenEnv
from collections import deque

#make environment
def make_env(seed=0):
    board = Board()
    bag = PieceBag(PIECE_LIBRARY, seed=seed)
    game = Game(board, bag)
    return TenTenEnv(game)

env = make_env(seed=0)

out = env.reset(seed=0)
obs, info = out if isinstance(out, tuple) else (out, {})
mask = env.action_mask()


#implemenent torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def obs_to_torch(obs):
    board = torch.as_tensor(obs["board"], dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1,10,10)
    hand_masks = torch.as_tensor(obs["hand_masks"], dtype=torch.float32, device=DEVICE)     # (3,5,5)
    return board, hand_masks

#Q network class
class QNet(nn.Module):
    def __init__(self, n_actions=300):
        super().__init__()
        self.board_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.hand_cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        board_out = 32 * 10 * 10
        hand_out  = 32 * 5 * 5
        self.head = nn.Sequential(
            nn.Linear(board_out + hand_out, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, board, hand_masks):
        if board.dim() == 3: board = board.unsqueeze(0)            # (B,1,10,10)
        if hand_masks.dim() == 3: hand_masks = hand_masks.unsqueeze(0)  # (B,3,5,5)
        xb = self.board_cnn(board)
        xh = self.hand_cnn(hand_masks)
        x = torch.cat([xb, xh], dim=1)
        return self.head(x)  # (B,300)

#action selection
def select_action(net, obs, mask, epsilon=0.1):
    legal = np.flatnonzero(mask)
    if len(legal) == 0:
        return None
    
    #implement prioritized relay
    if np.random.rand() < epsilon:
        return int(np.random.choice(legal))

    board, hand_masks = obs_to_torch(obs)
    with torch.no_grad():
        q = net(board, hand_masks).squeeze(0)  # (300,)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=q.device)
        q[~mask_t] = -1e9
        return int(torch.argmax(q).item())

#replay buffer
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done, next_mask):
        # store numpy arrays (lightweight, avoids torch memory issues)
        self.buf.append((
            obs["board"].copy(),
            obs["hand_masks"].copy(),
            int(action),
            float(reward),
            next_obs["board"].copy(),
            next_obs["hand_masks"].copy(),
            bool(done),
            next_mask.copy()  # (300,) bool
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return batch

    def __len__(self):
        return len(self.buf)


def batch_to_torch(batch):
    # batch is a list of tuples from ReplayBuffer
    boards = torch.as_tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=DEVICE)      # (B,10,10)
    hands  = torch.as_tensor(np.stack([b[1] for b in batch]), dtype=torch.float32, device=DEVICE)      # (B,3,5,5)
    acts   = torch.as_tensor([b[2] for b in batch], dtype=torch.int64, device=DEVICE)                  # (B,)
    rews   = torch.as_tensor([b[3] for b in batch], dtype=torch.float32, device=DEVICE)                # (B,)
    nboards= torch.as_tensor(np.stack([b[4] for b in batch]), dtype=torch.float32, device=DEVICE)      # (B,10,10)
    nhands = torch.as_tensor(np.stack([b[5] for b in batch]), dtype=torch.float32, device=DEVICE)      # (B,3,5,5)
    dones  = torch.as_tensor([b[6] for b in batch], dtype=torch.float32, device=DEVICE)                # (B,)
    nmask = torch.tensor(
        np.stack([b[7] for b in batch]).astype(np.bool_),
        dtype=torch.bool,
        device=DEVICE
    )


    boards = boards.unsqueeze(1)
    nboards = nboards.unsqueeze(1)

    return boards, hands, acts, rews, nboards, nhands, dones, nmask

#Prioritised relay
def dqn_update_per(online_net, target_net, optimizer, batch, weights, gamma=0.99):
    
    boards, hands, acts, rews, nboards, nhands, dones, nmask = batch_to_torch(batch)

    q_all = online_net(boards, hands)                         # (B,300)
    q_sa  = q_all.gather(1, acts.unsqueeze(1)).squeeze(1)     # (B,)

    with torch.no_grad():
        q_next_all = target_net(nboards, nhands)              # (B,300)
        q_next_all[~nmask] = -1e9
        max_next = q_next_all.max(dim=1).values               # (B,)
        y = rews + gamma * (1.0 - dones) * max_next           # (B,)

    td = (y - q_sa)                                           # (B,)

    # elementwise huber, then weight
    per_item = F.smooth_l1_loss(q_sa, y, reduction="none")    # (B,)
    loss = (weights * per_item).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
    optimizer.step()

    #return float(loss.item()), td.detach().cpu().numpy() #uniform
    return float(loss.item()), td.detach().abs().cpu() 

def dqn_update(online_net, target_net, optimizer, batch, gamma=0.99):
    boards, hands, acts, rews, nboards, nhands, dones, nmask = batch_to_torch(batch)

    # 1) Online net predicts Q(s,·)
    q_all = online_net(boards, hands)                 # (B,300)

    # 2) Pick Q(s,a) for the action actually taken
    q_sa = q_all.gather(1, acts.unsqueeze(1)).squeeze(1)  # (B,)

    # 3) Target: r + gamma * max_{legal a'} Q_target(s',a')
    with torch.no_grad():
        q_next_all = target_net(nboards, nhands)      # (B,300)
        q_next_all[~nmask] = -1e9                     # mask illegal
        max_next = q_next_all.max(dim=1).values       # (B,)
        y = rews + gamma * (1.0 - dones) * max_next   # (B,)

    # 4) Loss: Huber loss between prediction and target
    loss = F.smooth_l1_loss(q_sa, y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
    optimizer.step()

    return float(loss.item())

class PrioritizedReplayBuffer:
    def __init__(self, capacity=100_000, alpha=0.6, eps=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps

        self.buf = [None] * capacity
        self.priorities = np.zeros(capacity, dtype=np.float32)

        self.pos = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done, next_mask):
        data = (
            obs["board"].copy(),
            obs["hand_masks"].copy(),
            int(action),
            float(reward),
            next_obs["board"].copy(),
            next_obs["hand_masks"].copy(),
            bool(done),
            next_mask.copy()
        )

        self.buf[self.pos] = data

        # new samples get max priority so they will be learned soon
        max_p = self.priorities[:self.size].max() if self.size > 0 else 1.0
        self.priorities[self.pos] = max_p

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        assert self.size > 0

        # compute sampling probability
        p = self.priorities[:self.size] + self.eps
        p = p ** self.alpha
        #probs = p / p.sum() #uniform
        probs = (p / p.sum()).astype(np.float32)

        idxs = np.random.choice(self.size, batch_size, p=probs)
        batch = [self.buf[i] for i in idxs]

        # importance sampling weights
        weights = (self.size * probs[idxs]) ** (-beta)
        weights /= weights.max()

        weights = torch.tensor(weights.tolist(), dtype=torch.float32, device=DEVICE)
        #weights = torch.as_tensor(weights, dtype=torch.float32, device=DEVICE) #uniform

        return batch, idxs, weights

    def update_priorities(self, idxs, td_errors):
        
        td_errors = np.asarray(td_errors, dtype=np.float32)
        new_p = np.abs(td_errors) + self.eps
        self.priorities[idxs] = new_p

    def __len__(self):
        return self.size
    
def greedy_action(net, obs, mask):
    legal = np.flatnonzero(mask)
    if len(legal) == 0:
        return None
    board, hand_masks = obs_to_torch(obs)
    with torch.no_grad():
        q = net(board, hand_masks).squeeze(0)
        q[~torch.as_tensor(mask, dtype=torch.bool, device=q.device)] = -1e9
        return int(torch.argmax(q).item())
    
if __name__ == "__main__":
    WATCH_EVERY = 1e10       # watch every 10 episodes
    WATCH_MAX_STEPS = 300
    WATCH_DELAY = 0.05

    USE_PER = True #PER USE
    best_return = -float("inf")
    best_step = 0
    #Creates the 1010 environment: board + bag + game + gym wrapper.
    env = make_env(seed=0)

    online = QNet().to(DEVICE)
    target = QNet().to(DEVICE)
    best_model = QNet().to(DEVICE)
    target.load_state_dict(online.state_dict())
    best_model.load_state_dict(online.state_dict())

    lr = 1e-4
    optimizer = optim.Adam(online.parameters(), lr=lr)
    rb = ReplayBuffer(capacity=100_000)
    if USE_PER:
        rb = PrioritizedReplayBuffer(capacity=300_000, alpha=0.6)
    else:
        rb = ReplayBuffer(capacity=300_000)

    #NUM_EPISODES = 1000       # <- stop condition
    batch_size = 64
    gamma = 0.99
    ep = 0  #training purposes

    MAX_STEPS = 1_000_000   # total transitions to collect (tune this)
    #NUM_EPISODES = 0 
    #start_learning = 200
    #target_update_every = 500
    #epsilon_decay_steps = 2000

    start_learning = 10000       # wait until buffer has enough
    train_every = 4             # learn every N environment steps
    target_update_every = 5000  # copy online -> target periodically

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay_steps = 800000

    global_step = 0
    
    wandb.init(project="tenten-dqn", config={
        "lr": lr,
        "gamma": gamma,
        "use_per": USE_PER,
        "batch_size": batch_size,
        "target_update_every": target_update_every,
        "start_learning": start_learning,
    })


    while global_step < MAX_STEPS: #for training purpose
    #for ep in range(NUM_EPISODES):
        out = env.reset(seed=ep)
        obs, info = out if isinstance(out, tuple) else (out, {})
        mask = env.action_mask().astype(np.bool_)
        ep_return = 0.0
        done = False
        ep_steps = 0

        while not done and global_step < MAX_STEPS: #for actual training
        #while not done:
            # choose action (random early, greedy later)
            a = select_action(online, obs, mask, epsilon=epsilon)
            if a is None:
                done = True
                break

            next_obs, reward, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)
            next_mask = env.action_mask() if not done else np.zeros(300, dtype=bool)

            rb.add(obs, a, reward, next_obs, done, next_mask)

            obs = next_obs
            mask = next_mask
            ep_return += reward
            global_step += 1 #for training purpose
            ep_steps += 1

            if global_step % LOG_EVERY_STEPS == 0:
                now = time.time()
                steps_done = global_step - last_step
                dt = now - last_t
                sps = steps_done / max(dt, 1e-9)  # steps per second
                print(f"[speed] steps={global_step}  {sps:.1f} steps/sec")
                wandb.log({"perf/steps_per_sec": sps}, step=global_step)
                last_t = now
                last_step = global_step

            # epsilon schedule
            frac = min(1.0, global_step / epsilon_decay_steps)
            epsilon = max(epsilon_min, 1.0 - frac * (1.0 - epsilon_min))

            # learn
            if len(rb) >= start_learning and global_step % train_every == 0:
                if not USE_PER:
                    batch = rb.sample(batch_size)
                    loss = dqn_update(online, target, optimizer, batch, gamma=gamma)
                    wandb.log({"loss/td": loss}, step=global_step)

                else:
                    beta0, beta1 = 0.4, 1.0
                    frac = min(1.0, global_step / MAX_STEPS)
                    beta = beta0 + frac * (beta1 - beta0)

                    batch, idxs, weights = rb.sample(batch_size, beta=beta)
                    loss, td_errors = dqn_update_per(online, target, optimizer, batch, weights, gamma=gamma)
                    #rb.update_priorities(idxs, td_errors) uniform
                    rb.update_priorities(idxs, td_errors.tolist())
                    wandb.log({"loss/td": loss, "per/beta": beta}, step=global_step)

            # update target network
            if global_step % target_update_every == 0:
                target.load_state_dict(online.state_dict())
            
        print(f"episode={ep:03d} return={ep_return:.1f} eps={epsilon:.3f} buffer={len(rb)}")
        wandb.log(
            {
                "charts/episode_return": ep_return,
                "charts/episode_length": ep_steps,
                "charts/epsilon": epsilon,
                "charts/buffer": len(rb),
            },
            step=global_step,
        )
        
        if ep_return>best_return:
            best_return = ep_return
            best_step = global_step
            print(f"new best return={best_return} return={best_step}") 
            best_model.load_state_dict(online.state_dict())
            wandb.log({"charts/best_return": best_return}, step=global_step)

        # Watch one greedy game (prints ASCII each step)

        if ep % WATCH_EVERY == 0:
            env_watch = make_env(seed=123)
            out = env_watch.reset(seed=123)
            obs_w, info_w = out if isinstance(out, tuple) else (out, {})

            done_w = False
            steps = 0

            while not done_w and steps < WATCH_MAX_STEPS:
                mask_w = env_watch.action_mask().astype(np.bool_)

                a = greedy_action(online, obs_w, mask_w)
                if a is None:
                    break

                obs_w, r, terminated, truncated, info_w = env_watch.step(a)
                #env_watch.render()
                time.sleep(WATCH_DELAY)

                done_w = bool(terminated or truncated)
                steps += 1
        ep += 1

    print(f"highest return={best_return} return={best_step}") 
    torch.save(best_model.state_dict(), f"best_dqn_model_{best_return:.2f}.pt")
    wandb.finish()