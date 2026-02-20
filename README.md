# Getting Started

Clone Repo:
`git clone https://github.com/JaigonRyu/ten_ten_rl.git`

`cd ten_ten_rl`

Create Virtual Envrioment (using conda):

`conda create -n ten_ten_env python=3.10`

`conda activate ten_ten_env`

`pip install -r requirements.txt`

Code to run random baseline and random and greedy scripts from terminal:

(easiest to start with)
`python -m ten_ten_rl.scripts.random_play`

`python -m ten_ten_rl.scripts.greedy_play`



**Note**: I am running the commands in the terminal from a file path up. So I have ten_ten_rl in a DSC291 file. I am in the DSC291 file when I rum the commands. 


## File Path Layout

ten_ten_rl/
  core/
    __init__.py
    board.py
    bag.py
    pieces.py
    game.py
  env/
    __init__.py
    ten_ten_env.py
  scripts/
    greedy_play.py
    random_play.py
  README.md
  requirements.txt


## Core Folder

**pieces.py**

**Note**: Pieces are defined relative to the (0,0) coordinate and anchored there then it is moved to the correct area and placed in other functions.

- Defines the fixed piece set and how pieces are represented.

Piece: simple class storing:

- pid: string id (e.g., "T_up")

- cells: list of (dr, dc) offsets relative to top-left anchor

- height, width: bounding box

- n_blocks: number of occupied cells

Piece.as_array(): returns a tight bounding-box array for debugging.

Piece.to_mask(size=5): returns a fixed-size padded mask (for RL observations).

PIECE_LIBRARY: list of all pieces allowed in the game.

PIECE_BY_ID: convenience dict for lookup by id.

**board.py**

Board.grid: array of shape (size, size) with values {0,1}

Board.can_place(piece, r, c): checks:

- piece fits within bounds

- all target cells are currently empty

Board.place(piece, r, c):

- writes the piece’s blocks into the grid

- calls clear_full_lines()

- returns a dict like:

{
  "placed_blocks": ...,
  "cleared_rows": ...,
  "cleared_cols": ...,
  "cleared_blocks": ...
}

Board.clear_full_lines():

- finds any completely filled rows and columns

- clears them in one pass per move 

- counts cleared blocks correctly (intersection cells counted once)

Board.copy() / Board.to_ascii():

- utilities for search (greedy/minimax simulation) and debugging.

**bag.py** (We can add the adversary here!)

Random piece sampler

PieceBag(piece_library, seed=None, rng=None)

Samples with replacement (classic pool behavior).

draw_one() and draw_hand(k=3)

reseed(seed) for reproducible episodes (can remove later if needed)

**game.py** 

Controls turns, hand management, scoring, and terminal condition.

Holds:

- board

- hand (list of Pieces)

- score

- done

reset():

- clears board, draws new hand, resets score/done

step((piece_index, r, c)) **<- reward stored here**:

- validates action

- calls board.place(...)

- removes used piece from hand

- refills hand when empty

- sets done=True if no legal moves exist

- **computes reward and updates score** (we can change this as well)

has_any_moves() / legal_actions():

- scans for legal placements (useful for greedy/minimax/debugging)

## Env Folder

Defines action_space = Discrete(300) mapping:

- action = piece_index * 100 + row * 10 + col (basically a flatten)

Builds observation dict:

- "board": (10,10)

- "hand_masks": (3,5,5) -> **IMPORTANT**: Tuple (max number of pieces in the hand, grid height, gird width) So for example the hand_mask[0] holds the 5x5 mask of the dot piece and hand_mask[1] holds the next piece and so on

Note: It is five by five beacause that holds the largest piece.

- "hand_valid": (3,)

reset(seed=...): reseeds bag + resets game

step(action):

- decodes action → checks legality

- applies move using game.step(...)

- returns (obs, reward, terminated, truncated, info)

action_mask():

- eturns boolean array of length 300 of legal actions

## Scripts

ten_ten_rl/scripts/random_play.py

- Random baseline using env.action_mask() to avoid invalid actions.

- Runs multiple episodes and prints average score.

ten_ten_rl/scripts/greedy_play.py

For each legal move:

- simulate on board.copy()

- choose max immediate reward (blocks placed + line clears bonus)