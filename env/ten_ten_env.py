# ten_ten/envs/ten_ten_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TenTenEnv(gym.Env):
    """
    Gymnasium wrapper for 1010.

    Discrete action encoding (size = hand_size * N * N):
        action = piece_index * (N*N) + row * N + col

    Observation (dict):
      - board: (N, N) uint8
      - hand_masks: (hand_size, mask_size, mask_size) uint8
        (each piece padded to mask_size x mask_size; empty slots are all zeros)
    """

    def __init__(self, game, invalid_move_penalty=-5.0, mask_size=5, hand_size=3):
        super().__init__()
        self.game = game
        self.invalid_move_penalty = float(invalid_move_penalty)

        self.mask_size = int(mask_size)
        self.hand_size = int(hand_size)

        # Board size (supports progressive grid sizes later)
        self.N = int(self.game.board.size)

        # Action space: hand_size * N * N
        self.actions_per_piece = self.N * self.N
        self.action_dim = self.hand_size * self.actions_per_piece
        self.action_space = spaces.Discrete(self.action_dim)
        
        # Observation space
        # builds a gymnasium space using Box for the grid and hand mask
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0, high=1, shape=(self.N, self.N), dtype=np.uint8),
                "hand_masks": spaces.Box(
                    low=0, high=1, shape=(self.hand_size, self.mask_size, self.mask_size), dtype=np.uint8
                ),
            }
        )
  

    def _decode_action(self, action):
        a = int(action)
        pi = a // self.actions_per_piece
        rem = a % self.actions_per_piece
        r = rem // self.N
        c = rem % self.N
        return pi, r, c 

    def _encode_hand_masks(self):
        masks = np.zeros((self.hand_size, self.mask_size, self.mask_size), dtype=np.uint8)
        for i in range(min(self.hand_size, len(self.game.hand))):
            masks[i] = self.game.hand[i].to_mask(self.mask_size)
        return masks

    def _obs(self):
        return {
            "board": self.game.board.grid.copy().astype(np.uint8),
            "hand_masks": self._encode_hand_masks(),
        }

    def action_mask(self):
        """
        Boolean array of shape (action_dim,). True = legal.
        Useful for baselines
        """
        mask = np.zeros((self.action_dim,), dtype=bool)

        board = self.game.board
        hand = self.game.hand

        for pi in range(min(self.hand_size, len(hand))):
            piece = hand[pi]
            max_r = self.N - piece.height
            max_c = self.N - piece.width

            for r in range(max_r + 1):
                for c in range(max_c + 1):
                    if board.can_place(piece, r, c):
                        a = pi * self.actions_per_piece + r * self.N + c
                        mask[a] = True

        return mask
    
    #for stable baselines3 contrib
    def action_masks(self):
        return self.action_mask()

    #gym api stuff

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.game.bag.reseed(seed)

        self.game.reset()
        return self._obs(), {}

    def step(self, action):
        pi, r, c = self._decode_action(action)

        # invalid piece slot
        if pi < 0 or pi >= len(self.game.hand):
            return self._obs(), self.invalid_move_penalty, False, False, {"invalid": True}

        piece = self.game.hand[pi]

        # illegal placement
        if not self.game.board.can_place(piece, r, c):
            return self._obs(), self.invalid_move_penalty, False, False, {"invalid": True}

        # legal
        _, reward, done, info_core = self.game.step((pi, r, c))
        obs = self._obs()

        terminated = bool(done)
        truncated = False

        return obs, float(reward), terminated, truncated, info_core


    def render(self):
        # Optional: simple ASCII render
        print(self.game.board.to_ascii())
        print("Hand:", [p.pid for p in self.game.hand])
        print("Score:", self.game.score)

    def close(self):
        pass


