
from typing import Iterable, List, Sequence, Tuple, Dict
import numpy as np

class Piece:

    """
    Piece is a class that contains all the information for a specific piece 
    """

    def __init__(self, pid, cells):

        if not cells:
            raise ValueError("Invalid Piece")
        
        self.pid = pid


        seen = set()
        cleaned = []

        for row, col in cells:
            
            if (row, col) in seen:
                raise ValueError("Duplicate cell")
            
            seen.add((row,col))
            cleaned.append((row,col))

        
        min_row = min(row for row, _ in cleaned)
        min_col = min(col for _, col in cleaned)

        norm = [(row - min_row, col-min_col) for row, col in cleaned]
        norm.sort()

        self.cells = norm
        self.n_blocks = len(self.cells)

        max_row = max(row for row, _ in self.cells)
        max_col = max(col for _, col in self.cells)

        self.height = max_row + 1
        self.width = max_col + 1

    def as_array(self):

        arr = np.zeros((self.height, self.width))
        for row, col in self.cells:
            arr[row, col] = 1
        return arr
    
    def to_mask(self, size=5):
        """
        Return a size x size uint8 mask (padded), top-left aligned.
        Useful for RL observations.
        """
        if self.height > size or self.width > size:
            raise ValueError(
                f"Piece {self.pid} ({self.height}x{self.width}) won't fit in {size}x{size}."
            )
        mask = np.zeros((size, size), dtype=np.uint8)
        for r, c in self.cells:
            mask[r, c] = 1
        return mask

    def __repr__(self):
        return f"Piece(pid={self.pid!r}, n_blocks={self.n_blocks}, size={self.height}x{self.width})"
    

## Library of all the pieces that we currently have in play. We can edit and change them later if we want.
def make_library():
    
    pieces = []

    def add(pid, cells):
        pieces.append(Piece(pid, cells))

    # Single
    add("dot", [(0, 0)])

    # Lines (2..5), horizontal + vertical
    add("line2_h", [(0, 0), (0, 1)])
    add("line2_v", [(0, 0), (1, 0)])

    add("line3_h", [(0, 0), (0, 1), (0, 2)])
    add("line3_v", [(0, 0), (1, 0), (2, 0)])

    add("line4_h", [(0, 0), (0, 1), (0, 2), (0, 3)])
    add("line4_v", [(0, 0), (1, 0), (2, 0), (3, 0)])

    add("line5_h", [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)])
    add("line5_v", [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])

    # Squares
    add("sq2", [(0, 0), (0, 1),
                (1, 0), (1, 1)])

    add("sq3", [(0, 0), (0, 1), (0, 2),
                (1, 0), (1, 1), (1, 2),
                (2, 0), (2, 1), (2, 2)])

    # L (3 blocks) - 4 orientations
    add("L3_a", [(0, 0), (1, 0), (1, 1)])
    add("L3_b", [(0, 1), (1, 0), (1, 1)])
    add("L3_c", [(0, 0), (0, 1), (1, 0)])
    add("L3_d", [(0, 0), (0, 1), (1, 1)])

    # L (4 blocks) - 4 orientations
    add("L4_a", [(0, 0), (1, 0), (2, 0), (2, 1)])
    add("L4_b", [(0, 1), (1, 1), (2, 1), (2, 0)])
    add("L4_c", [(0, 0), (0, 1), (1, 0), (2, 0)])
    add("L4_d", [(0, 0), (0, 1), (1, 1), (2, 1)])


    return pieces



PIECE_LIBRARY = make_library()
PIECE_BY_ID = {p.pid: p for p in PIECE_LIBRARY}

def all_pieces():
    return list(PIECE_LIBRARY)