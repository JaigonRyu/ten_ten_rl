# ten_ten/core/game.py

class Game:
    """
      - Holds board + current hand
      - Applies actions (piece_index, r, c)
      - Clears lines via Board.place()
      - Refill hand when empty
      - Detect terminal when no legal moves exist
    """

    def __init__(self, board, bag, hand_size=3):
        self.board = board
        self.bag = bag
        self.hand_size = hand_size

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
            raise RuntimeError("Cannot step() because game is already done. Call reset().")

        piece_index, r, c = action

    
        if piece_index < 0 or piece_index >= len(self.hand):
            # Minimal behavior: treat as invalid action and end or penalize.
            # just raise to catch bugs early. Env wrapper can handle penalties later.
            raise ValueError(f"Invalid piece_index {piece_index} for hand size {len(self.hand)}.")

        piece = self.hand[piece_index]

        if not self.board.can_place(piece, r, c):
           
            raise ValueError(f"Illegal move: {piece.pid} at ({r}, {c})")

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
        # We can change this to see if different rewards are better.
        reward = place_info["placed_blocks"]
        reward += 10 * (place_info["cleared_rows"] + place_info["cleared_cols"])

        self.score += reward

        info = {
            "piece_id": piece.pid,
            "place_info": place_info,
        }

        return self.get_state(), reward, self.done, info
