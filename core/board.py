import numpy as np

class Board:
    """
    A class to represent the board of our game which is a numpy grid

    """

    def __init__(self, size=10):

        """
        Initialize board with size

        Parameters:
            size (int): size is the number of grid cells denoted nxn so size =10 gives a board of size 10x10

        """

        self.size = size
        self.grid = np.zeros((size, size))

    def reset(self):
        """
        Resets the board by filling it with zeros. 
        """
        self.grid.fill(0)

    def copy(self):

        """
        Creates a copy of the board to use 

        Returns:
            board_copy (class): Returns a copy of the board.
        """

        board_copy = Board(self.size)
        board_copy.grid = self.grid.copy()
        return board_copy
        
    def in_bounds(self, row, col):

        """
        in_bounds checks to see if the row of column value is within the bounds of the board size.
        It shoudl not be less than zero or greater than the self.size
 

        Parameters:
            row (int): integer representing row value
            col (int): integer representing col value

        Returns:
            draw_hand (list): Returns a list of k draws for the game to use each turn.
        """

        return 0 <= row < self.size and 0 <= col < self.size
    
    def can_place(self, piece, top_row, top_col):

        """
        Can place checks to see if the block is valid to be placed. It checks to make sure it does not exceed the boundaries of the grid
        It then checks if there is any other piece in the way of placement.
 

        Parameters:
            piece (class): class repreenting piece
            top_row (int): integer representing the top row 
            top_row (int): integer representing  the top col

        Returns:
            True/False (Boolean): Returns a boolean value of whether the piece can be placed. 
        """

        if top_row < 0 or top_col < 0:
            return False
        
        if top_row + piece.height > self.size:
            return False
        
        if top_col + piece.width > self.size:
            return False
        
        for dr, dc in piece.cells:

            r = top_row + dr
            c = top_col + dc

            if self.grid[r,c] == 1:
                return False
            
        return True
    
    def place(self, piece, top_row, top_col):

        """
        This function does the actual placing of the blocks on the board. It also calls funciton to clear any lines.
 

        Parameters:
            piece (class): class repreenting piece
            top_row (int): integer representing the top row 
            top_row (int): integer representing  the top col

        Returns:
            Dictionary (Dict.): Returns a dictionary of the cleared blocks, rows, cols, and placed blocks.
        """

        if not self.can_place(piece, top_row, top_col):
            raise ValueError("Bad Place")
        
        for dr, dc in piece.cells:
            self.grid[top_row + dr, top_col + dc] = 1

        
        cleared_rows, cleared_cols, cleared_blocks = self.clear_full_lines()

        return {
            "placed_blocks": piece.n_blocks,
            "cleared_rows": cleared_rows,
            "cleared_cols": cleared_cols,
            "cleared_blocks": cleared_blocks,
        }
    
    def clear_full_lines(self):

        """
        This function checks for full rows and columns. If none it returns zeros. If there is it creates a clear mask.
        The clear mask is then used in self.grid to clear that row/col. 

        Returns:
            Three values (integers): first val is the num of full rows cleared, second is full columns cleared, number of cleared blocks total
            Used for reward shaping.
        """

        full_rows = np.where(self.grid.sum(axis=1) == self.size)[0]
        full_cols = np.where(self.grid.sum(axis=0) == self.size)[0]

        if len(full_rows) == 0 and len(full_cols) == 0:
            return 0, 0, 0

        clear_mask = np.zeros_like(self.grid, dtype=bool)
        if len(full_rows) > 0:
            clear_mask[full_rows, :] = True
        if len(full_cols) > 0:
            clear_mask[:, full_cols] = True

        cleared_blocks = int(self.grid[clear_mask].sum())

        self.grid[clear_mask] = 0

        return int(len(full_rows)), int(len(full_cols)), cleared_blocks
    
    # Helper functions below
    def count_filled(self):
        return int(self.grid.sum())

    def is_full_row(self, r):
        return int(self.grid[r, :].sum()) == self.size

    def is_full_col(self, c):
        return int(self.grid[:, c].sum()) == self.size

    def __repr__(self):
        return f"Board(size={self.size}, filled={self.count_filled()})"

    # ascii render that is useful for visuals
    def to_ascii(self):
        
        lines = []
        for r in range(self.size):
            row = "".join("#" if x else "." for x in self.grid[r])
            lines.append(row)
        return "\n".join(lines)

                