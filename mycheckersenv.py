import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary
from gymnasium.utils import seeding

from pettingzoo import AECEnv


class CustomEnvironment(AECEnv):
    metadata = {
        "name": "custom_environment_v0",
        "render_modes": ["human"],
    }

    BOARD_SIZE = 6
    EMPTY = 0
    P1_MAN = 1
    P1_KING = 2
    P2_MAN = -1
    P2_KING = -2
    
    def __init__(self, render_mode=None, max_moves=200):
        self.possible_agents = ["player_0", "player_1"]
        self.agent_name_mapping = {"player_0": 0, "player_1": 1}
        self.render_mode = render_mode
        self.max_moves = max_moves

        self._action_tuples = self._build_action_tuples()
        self._action_to_idx = {a: i for i, a in enumerate(self._action_tuples)}

        # Define observation space
        board_space = Box(low=-2, high=2, shape=(self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        self.observation_spaces = {
            agent: Dict(
                {
                    "board": board_space,
                    "action_mask": MultiBinary(len(self._action_tuples)),
                    "current_player": Discrete(2),
                }
            )
            for agent in self.possible_agents
        }

        # Define action space
        self.action_spaces = {
            agent: Discrete(len(self._action_tuples)) for agent in self.possible_agents
        }

        # Initialize other attributes
        self.board = None
        self.agents = []
        self.forced_capture_piece = None
        self.num_moves = 0
        self.np_random = None
        self.np_random_seed = None

        # Reset the environment
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        self.agent_selection = self.possible_agents[0]

    # Helper methods for game logic: move generation, move execution, win condition checking, etc.
    def _build_action_tuples(self):
        actions = []
        for start_row in range(self.BOARD_SIZE):
            for start_col in range(self.BOARD_SIZE):

                # Check all valid moves - dr, dc are the row and column deltas for the move
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-2, -2), (-2, 2), (2, -2), (2, 2)]:
                    
                    end_row, end_col = start_row + dr, start_col + dc     
                    if 0 <= end_row < self.BOARD_SIZE and 0 <= end_col < self.BOARD_SIZE:
                        actions.append((start_row, start_col, end_row, end_col))
        return actions

    # Game logic helper methods
    # Check for king, piece ownership, opponent pieces
    def _is_king(self, piece):
        return abs(piece) == 2

    def _belongs_to(self, piece, agent):
        if agent == "player_0":
            return piece > 0
        return piece < 0

    def _is_opponent(self, piece, agent):
        if piece == self.EMPTY:
            return False
        return not self._belongs_to(piece, agent)

    # Move generation methods
    # Forward direction depends on the player
    def _forward_dirs(self, agent):
        # player_0 starts at bottom and moves up, player_1 starts at top and moves down.
        return [(-1, -1), (-1, 1)] if agent == "player_0" else [(1, -1), (1, 1)]

    # Move direction depends on the piece
    def _move_dirs(self, piece, agent):
        if self._is_king(piece):
            return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        return self._forward_dirs(agent)

    # Capture generation and simple move generation for a piece at (row, col)
    def _capture_moves_from(self, row, col, agent):
        piece = self.board[row, col]

        # If the piece is empty or doesn't belong to the agent, it can't capture.
        if piece == self.EMPTY or not self._belongs_to(piece, agent):
            return []

        # Check for captures in all valid directions
        captures = []
        for dr, dc in self._move_dirs(piece, agent):
            mid_r, mid_c = row + dr, col + dc
            end_r, end_c = row + 2 * dr, col + 2 * dc
            if 0 <= end_r < self.BOARD_SIZE and 0 <= end_c < self.BOARD_SIZE: # Ensure the landing square is on the board
                if self._is_opponent(self.board[mid_r, mid_c], agent) and self.board[end_r, end_c] == self.EMPTY:
                    captures.append((row, col, end_r, end_c))
        return captures

    # Simple moves are only allowed if there are no captures available for any piece of the agent.
    def _simple_moves_from(self, row, col, agent):
        piece = self.board[row, col]
        if piece == self.EMPTY or not self._belongs_to(piece, agent):
            return []

        # Simple moves are one step in a valid direction to an empty square. No jumping allowed.
        moves = []
        # Check for simple moves in all valid directions
        for delta_row, delta_col in self._move_dirs(piece, agent):  # dr, dc are the row and column deltas for the move
            new_row, new_col = row + delta_row, col + delta_col     # new row and column after the move
            if 0 <= new_row < self.BOARD_SIZE and 0 <= new_col < self.BOARD_SIZE and self.board[new_row, new_col] == self.EMPTY: # Ensure the landing square is on the board
                moves.append((row, col, new_row, new_col))
        return moves

    # Generate all legal moves for a given agent
    def _all_legal_moves(self, agent):
        captures = []
        simples = []

        # If there is a forced capture piece, only generate captures for that piece.
        if self.forced_capture_piece is not None:
            fr, fc = self.forced_capture_piece
            return self._capture_moves_from(fr, fc, agent)

        # Otherwise, check all pieces for captures first. If any captures are found, only those are legal.
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if self._belongs_to(self.board[row, col], agent):
                    captures.extend(self._capture_moves_from(row, col, agent))

        # If no captures are found, then generate simple moves for all pieces.
        if captures:
            return captures

        # No captures, so generate simple moves.
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if self._belongs_to(self.board[row, col], agent):
                    simples.extend(self._simple_moves_from(row, col, agent))

        return simples

    # Promotion occurs when a man reaches the opposite end of the board. Check is made after a move action.
    def _promote_if_needed(self, row, col):
        piece = self.board[row, col]
        if piece == self.P1_MAN and row == 0:
            self.board[row, col] = self.P1_KING
        elif piece == self.P2_MAN and row == self.BOARD_SIZE - 1:
            self.board[row, col] = self.P2_KING

    # Count the number of pieces for a given agent
    def _piece_count(self, agent):
        if agent == "player_0":
            return int(np.sum(self.board > 0))
        return int(np.sum(self.board < 0))

    # Retrieve the other agent
    def _other_agent(self, agent):
        return "player_1" if agent == "player_0" else "player_0"

    # Determine the winner based on piece count and available moves. Called after each move to check for game end conditions.
    def _get_winner(self):
        # retrieve piece counts
        p0_pieces = self._piece_count("player_0")
        p1_pieces = self._piece_count("player_1")
        
        # check for end conditions
        if p0_pieces == 0 and p1_pieces == 0: # draw
            return None
        if p0_pieces == 0:                    # player 1 wins
            return "player_1"
        if p1_pieces == 0:                    # player 0 wins
            return "player_0"

        # check for stalemate
        p0_moves = self._all_legal_moves("player_0")
        p1_moves = self._all_legal_moves("player_1")
        if len(p0_moves) == 0 and len(p1_moves) == 0:  # draw
            return None 
        if len(p0_moves) == 0:                         # player 1 wins
            return "player_1"
        if len(p1_moves) == 0:                         # player 0 wins
            return "player_0"

        return None         # game is not over

    # Retrieve the action mask for a given agent
    def _action_mask(self, agent):
        # If the agent is not the current agent, then return an empty mask
        mask = np.zeros(len(self._action_tuples), dtype=np.int8)
        if agent not in self.agents:
            return mask
        if agent != self.agent_selection:
            return mask
        if self.terminations.get(agent, False) or self.truncations.get(agent, False):
            return mask

        # Otherwise, generate the mask
        legal = self._all_legal_moves(agent)
        for move in legal:
            mask[self._action_to_idx[move]] = 1
        return mask

    def observe(self, agent):
        current_agent = self.agent_selection
        if current_agent not in self.agent_name_mapping:
            current_agent = self.possible_agents[0]

        return {
            "board": self.board.copy(),
            "action_mask": self._action_mask(agent),
            "current_player": self.agent_name_mapping[current_agent],
        }

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)

        self.agents = self.possible_agents[:]
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)

        # Initialize a 6x6 checkers setup on dark squares.
        for row in [0, 1]:
            for col in range(self.BOARD_SIZE):
                if (row + col) % 2 == 1:
                    self.board[row, col] = self.P2_MAN

        # Player 0's pieces are at the bottom, player 1's pieces are at the top. Only dark squares are used for pieces.
        for r in [4, 5]:
            for c in range(self.BOARD_SIZE):
                if (r + c) % 2 == 1:
                    self.board[r, c] = self.P1_MAN

        # Player 0 starts first
        self.agent_selection = "player_0"
        self.forced_capture_piece = None
        self.num_moves = 0
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

        # If render_mode is human, render the initial state after reset
        if self.render_mode == "human":
            self.render()

    # Perform one step of the game
    def step(self, action):
        if not self.agents:
            return

        acting_agent = self.agent_selection
        if self.terminations[acting_agent] or self.truncations[acting_agent]:
            self._was_dead_step(action)
            return

        self._cumulative_rewards[acting_agent] = 0.0
        self._clear_rewards()

        other_agent = self._other_agent(acting_agent)

        # Validate the action for the current agent.
        try:
            action_int = int(action)
        except (TypeError, ValueError):
            action_int = -1

        if action is None or action_int < 0 or action_int >= len(self._action_tuples):
            # Missing or invalid action is treated as an illegal move.
            self.rewards[acting_agent] = -1.0
            self.rewards[other_agent] = 1.0
            self.terminations = {agent: True for agent in self.possible_agents}
            self.agent_selection = other_agent
            self._accumulate_rewards()
            if self.render_mode == "human":
                self.render()
            return

        action = action_int

        # Apply the action to the game state.
        move = self._action_tuples[action]
        legal_moves = self._all_legal_moves(acting_agent)
        if move not in legal_moves:
            self.rewards[acting_agent] = -1.0
            self.rewards[other_agent] = 1.0
            self.terminations = {agent: True for agent in self.possible_agents}
            self.agent_selection = other_agent
            self._accumulate_rewards()
            if self.render_mode == "human":
                self.render()
            return

        # Apply the move
        start_row, start_col, end_row, end_col = move
        piece = self.board[start_row, start_col]
        self.board[start_row, start_col] = self.EMPTY
        self.board[end_row, end_col] = piece

        # Check if the move was a capture and remove the captured piece if so.
        was_capture = abs(end_row - start_row) == 2
        if was_capture:
            mr, mc = (start_row + end_row) // 2, (start_col + end_col) // 2
            self.board[mr, mc] = self.EMPTY

        self._promote_if_needed(end_row, end_col)

        self.num_moves += 1

        # Multi-jump: if more captures are available with the moved piece, the same agent plays again.
        if was_capture:
            follow_up_captures = self._capture_moves_from(end_row, end_col, acting_agent)
            if follow_up_captures:
                self.forced_capture_piece = (end_row, end_col)
                next_agent = acting_agent
            else:
                self.forced_capture_piece = None
                next_agent = other_agent
        else:
            self.forced_capture_piece = None
            next_agent = other_agent
        
        # Check if the game is over
        winner = self._get_winner()
        if winner is not None:
            self.rewards[winner] = 1.0
            self.rewards[self._other_agent(winner)] = -1.0
            self.terminations = {agent: True for agent in self.possible_agents}

        # Check if the game has reached the maximum number of moves
        if self.num_moves >= self.max_moves and winner is None:
            self.truncations = {agent: True for agent in self.possible_agents}

        self.agent_selection = next_agent
        self._accumulate_rewards()

        # If render_mode is human, render the updated state after the step
        if self.render_mode == "human":
            self.render()

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        piece_to_char = {
            self.EMPTY: ".",
            self.P1_MAN: "r",
            self.P1_KING: "R",    # promotion for player 0 is uppercase R
            self.P2_MAN: "b", 
            self.P2_KING: "B",    # promotion for player 1 is uppercase B
        }

        header = "  " + " ".join(str(c) for c in range(self.BOARD_SIZE))
        rows = [header]
        for r in range(self.BOARD_SIZE):
            row_chars = [piece_to_char[int(self.board[r, c])] for c in range(self.BOARD_SIZE)]
            rows.append(f"{r} " + " ".join(row_chars))

        live = not all(
            self.terminations[a] or self.truncations[a] for a in self.possible_agents
        )
        turn_info = f"Turn: {self.agent_selection}" if live else "Game over"
        print("\n".join(rows))
        print(turn_info)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]


def env(render_mode=None, max_moves=200):
    return CustomEnvironment(render_mode=render_mode, max_moves=max_moves)