import numpy as np

class PieceBag:
    """
    Docstring for PieceBag:

    Piecebag is how we will store all of the pieces that we need
    and how we will draw pieces

    - init has the initialization of the library and setting of random seed

    - draw one: draws a piece randomly and returns that piece using dict

    - drawhand calls draw one three times

    - reseed allows us to change the random seed per run 
    """

    def __init__(self, piece_library, seed=None, rng=None):

        """Initialzie the piece bag class that contains current pieces for the game

        Parameters:
            piece_library (list): A list of piece classes with each unique piece and thier associated methods

        """

        self.piece_library = list(piece_library)

        if rng is not None:
            self.rng = rng

        else:
            self.rng = np.random.default_rng(seed)

    def draw_one(self):

        """Return one random draw from the piecebag

        Returns:
            Piece (class): Returns piece class from bag that we will use in the game draw
        """

        index = self.rng.integers(0, len(self.piece_library))
        return self.piece_library[index]
        

    def draw_hand(self, k=3):

        """Return the sum of two decimal numbers in binary digits.

        Parameters:
            k (int): k is the number of draws per hand. 

        Returns:
            draw_hand (list): Returns a list of k draws for the game to use each turn.
        """
        return [self.draw_one() for _ in range(k)]
    
    def reseed(self, seed):
        self.rng = np.random.default_rng(seed)
    