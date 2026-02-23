# ten_ten/core/game.py

import numpy as np


class Game:
    """
    - Holds board + current hand
    - Applies actions (piece_index, r, c)
    - Clears lines via Board.place()
    - Refill hand when empty
    - Detect terminal when no legal moves exist
    """

    def __init__(self, board, bag, hand_size=3, hole_bonus=0.5, bump_bonus=0.1):
        self.board = board
        self.bag = bag
        self.hand_size = hand_size
        # Shaping weights; positive means reducing holes/bumpiness yields positive reward
        self.hole_bonus = float(hole_bonus)
        self.bump_bonus = float(bump_bonus)

        self.hand = []
        self.score = 0
        self.done = False

    def reset(self):
        self.board.reset()
        self.hand = self.bag.draw_hand(self.hand_size)
        self.score = 0
        self.done = False
        return self.get_state()

    def get_state(self):

        return {
            "board": self.board.grid.copy(),
            "hand_ids": [p.pid for p in self.hand],
            "score": self.score,
            "done": self.done,
        }

    def is_hand_empty(self):
        return len(self.hand) == 0

    def refill_hand_if_needed(self):
        if self.is_hand_empty():
            self.hand = self.bag.draw_hand(self.hand_size)

    def has_any_moves(self):

        size = self.board.size
        for piece in self.hand:

            max_r = size - piece.height
            max_c = size - piece.width
            for r in range(max_r + 1):
                for c in range(max_c + 1):
                    if self.board.can_place(piece, r, c):
                        return True
        return False

    def legal_actions(self):
        """
        Returns a list of all legal actions as tuples:
          (piece_index, r, c)
        This can be expensive but is fine for a baseline. We can change it later if it is an issue
        """
        actions = []
        size = self.board.size

        for i, piece in enumerate(self.hand):
            max_r = size - piece.height
            max_c = size - piece.width
            for r in range(max_r + 1):
                for c in range(max_c + 1):
                    if self.board.can_place(piece, r, c):
                        actions.append((i, r, c))

        return actions

    def step(self, action):
        """
        action: (piece_index, r, c)

        Returns (state, reward, done, info)
        """
        if self.done:
            raise RuntimeError(
                "Cannot step() because game is already done. Call reset()."
            )

        piece_index, r, c = action

        if piece_index < 0 or piece_index >= len(self.hand):
            # Minimal behavior: treat as invalid action and end or penalize.
            # just raise to catch bugs early. Env wrapper can handle penalties later.
            raise ValueError(
                f"Invalid piece_index {piece_index} for hand size {len(self.hand)}."
            )

        piece = self.hand[piece_index]

        if not self.board.can_place(piece, r, c):

            raise ValueError(f"Illegal move: {piece.pid} at ({r}, {c})")

        # Snapshot before move for shaping
        grid_before = self.board.grid.copy()

        # Apply move on board
        place_info = self.board.place(piece, r, c)

        # Remove used piece from hand
        self.hand.pop(piece_index)

        # Refill hand if needed
        self.refill_hand_if_needed()

        # Check terminal
        if not self.has_any_moves():
            self.done = True

        # Simple score / reward:

        # - Reward for placing blocks
        # - Bonus for clearing lines (rows+cols)
        reward = place_info["placed_blocks"]
        clear_bonus = place_info["cleared_rows"] + place_info["cleared_cols"]
        reward += 10 * clear_bonus

        # Shaping: reward reductions in holes/bumpiness (potential-based delta)
        holes_before, bump_before = self._board_holes_bumpiness(grid_before)
        holes_after, bump_after = self._board_holes_bumpiness(self.board.grid)
        reward += self.hole_bonus * (holes_before - holes_after)
        reward += self.bump_bonus * (bump_before - bump_after)

        self.score += reward
        reward += 10 * clear_bonus

        info = {
            "piece_id": piece.pid,
            "place_info": place_info,
            "score": self.score,
            "shaping": {
                "holes_before": holes_before,
                "holes_after": holes_after,
                "bump_before": bump_before,
                "bump_after": bump_after,
            },
        }

        return self.get_state(), reward, self.done, info

    def _board_holes_bumpiness(self, grid):
        # grid: (N, N) uint8/bool
        N = grid.shape[0]
        holes = 0
        heights = []
        for c in range(N):
            col = grid[:, c]
            filled = np.where(col != 0)[0]
            if filled.size == 0:
                heights.append(0)
                continue
            top = filled[0]
            height = N - top
            heights.append(height)
            holes += np.count_nonzero(col[top + 1 :] == 0)
        bumpiness = 0
        for i in range(N - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return holes, bumpiness
