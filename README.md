# Custom Checkers

This environment is a two-player, turn-based checkers-style game implemented with the PettingZoo AEC API.

## Import

```python
from mycheckersenv import CustomEnvironment, env
```

## Environment Summary

| Item | Value |
| --- | --- |
| Actions | Discrete |
| AEC API | Yes |
| Manual Control | No |
| Agents | `['player_0', 'player_1']` |
| Number of Agents | 2 |
| Action Shape | (1) |
| Action Values | Discrete(164) |
| Observation Type | Dict(`board`, `action_mask`, `current_player`) |
| Board Shape | (6, 6) |
| Action Mask Shape | (164) |
| Current Player Shape | Scalar (`Discrete(2)`) |

## Description

Custom Checkers is played on a 6x6 board using dark squares for piece movement. Each player starts with 6 men:

- `player_0` starts on rows 4 and 5 and moves upward.
- `player_1` starts on rows 0 and 1 and moves downward.

Pieces move diagonally:

- Men move one step forward diagonally.
- Kings move one step diagonally in any direction.
- Captures are diagonal jumps over an opponent piece into an empty square.

The game ends when a player has no pieces, has no legal moves, or when an illegal action is played. A game can also be truncated by `max_moves`.

## Observation Space

Each agent observes a dictionary:

- `board`: `Box(low=-2, high=2, shape=(6, 6), dtype=np.int8)`
- `action_mask`: `MultiBinary(164)`
- `current_player`: `Discrete(2)`

Board encoding:

| Value | Meaning |
| --- | --- |
| -2 | `player_1` king |
| -1 | `player_1` man |
| 0 | empty |
| 1 | `player_0` man |
| 2 | `player_0` king |

`current_player` is the index of the agent whose turn it is (`0` for `player_0`, `1` for `player_1`).

## Legal Actions Mask

Legal actions are provided via `action_mask` in each observation.

- The mask is binary, length 164.
- `1` means legal, `0` means illegal.
- For all non-acting agents, the mask is all zeros.

Forced capture is implemented:

- If any capture is available, only capture actions are legal.
- During a multi-jump sequence, only captures by the same moved piece are legal.

## Action Space

The action space is `Discrete(164)`.

Each action index maps to a tuple `(start_row, start_col, end_row, end_col)` generated from all in-bounds diagonal displacements:

- one-step diagonals: `(-1, -1), (-1, 1), (1, -1), (1, 1)`
- two-step diagonals: `(-2, -2), (-2, 2), (2, -2), (2, 2)`

Not every indexed action is legal in every state. Legality is defined by the current board and enforced through `action_mask`.

## Rewards

- Win: `+1` for winner, `-1` for loser.
- Loss: `-1` for loser, `+1` for winner.
- Draw/truncation: `0` for both agents.

Draw-like outcomes in this implementation return no winner and keep rewards at zero unless an illegal move occurred.

## Illegal Actions

If the acting agent provides a missing, out-of-range, or otherwise illegal action:

- acting agent reward: `-1`
- opponent reward: `+1`
- both agents are terminated immediately

## End Conditions

The episode terminates if any of the following occurs:

- one player has no remaining pieces
- one player has no legal moves
- an illegal action is taken

The episode is truncated when:

- `num_moves >= max_moves` and no winner has been determined

## Usage

### AEC API Example

```python
import numpy as np
from mycheckersenv import env

game = env(render_mode="human", max_moves=200)
game.reset(seed=42)

for agent in game.agent_iter():
	observation, reward, termination, truncation, info = game.last()
	if termination or truncation:
		action = None
	else:
		legal_actions = np.flatnonzero(observation["action_mask"])
		action = int(np.random.choice(legal_actions)) if legal_actions.size else 0
	game.step(action)
```

## API

### Class

`class CustomEnvironment(AECEnv)`

### Convenience Constructors

- `env(render_mode=None, max_moves=200)` returns `CustomEnvironment`

### Core Methods

- `reset(seed=None, options=None)`
- `step(action)`
- `render()`
- `observation_space(agent)`
- `action_space(agent)`

### Notes

- `render_mode="human"` prints an ASCII board.
- Promotion occurs automatically when a man reaches the opposite end row.
- Multi-jump capture turns are handled automatically by keeping the same current agent.
