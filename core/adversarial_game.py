import copy

from ten_ten_rl.core.game import Game
from ten_ten_rl.core.pieces import PIECE_LIBRARY


class AdversarialGame(Game):
    """
    Game variant where the "bag" is adversarial: after each turn the next hand
    is filled by picking pieces that minimize the player's best immediate score
    (a one-ply minimax: min over piece choice, max over that piece's best placement).
    """

    def __init__(self, board, bag, hand_size=3, line_bonus=10):
        super().__init__(board, bag, hand_size=hand_size)
        self.line_bonus = line_bonus
        self.piece_library = list(PIECE_LIBRARY)

    def reset(self):
        self.board.reset()
        self.hand = []
        self.score = 0
        self.done = False
        self._fill_hand_adversarial()
        return self.get_state()

    def refill_hand_if_needed(self):
        if self.is_hand_empty():
            self._fill_hand_adversarial()

    def _fill_hand_adversarial(self):
        while len(self.hand) < self.hand_size:
            worst_piece = self._pick_worst_piece_for_board()
            self.hand.append(worst_piece)

    def _pick_worst_piece_for_board(self):
        """
        For each candidate piece, compute the agent's best immediate score if
        that piece were drawn; choose the piece that minimizes this best score.
        """
        worst_score = float("inf")
        worst_piece = self.piece_library[0]

        for piece in self.piece_library:
            best_for_piece = self._best_immediate_score(piece)
            if best_for_piece < worst_score:
                worst_score = best_for_piece
                worst_piece = piece

        return worst_piece

    def _best_immediate_score(self, piece):
        """
        Simulate all legal placements of `piece` on the current board.
        Return the max immediate reward achievable with that piece.
        If no legal placement, return -inf to make such pieces least preferable.
        """
        board = self.board
        N = board.size
        best = -float("inf")
        max_r = N - piece.height
        max_c = N - piece.width

        for r in range(max_r + 1):
            for c in range(max_c + 1):
                if not board.can_place(piece, r, c):
                    continue
                b2 = board.copy()
                info = b2.place(piece, r, c)
                value = info["placed_blocks"] + self.line_bonus * (
                    info["cleared_rows"] + info["cleared_cols"]
                )
                if value > best:
                    best = value

        if best == -float("inf"):
            return -float("inf")
        return best
