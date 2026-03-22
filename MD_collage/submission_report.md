# CS272 PA2 - Custom Checkers Environment Report

## 1) Public GitHub Repository

Project repository (clickable):

[https://github.com/cubinCheese/PettingZoo_Checkers_Game](https://github.com/cubinCheese/PettingZoo_Checkers_Game)

## 2) Sample Run with Trained Agents (Board State Progression)

Source: trained_sample_run.log

```text
Loaded model from ac_selfplay_weights.npz
============================================================
Episode 1/1000
Epsilon: 0.200
  0 1 2 3 4 5
0 . b . b . b
1 b . b . b .
2 . . . . . .
3 . . . . . .
4 . r . r . r
5 r . r . r .
Turn: player_0

  0 1 2 3 4 5
0 . b . b . b
1 b . b . b .
2 . . . . . .
3 . . . . r .
4 . r . . . r
5 r . r . r .
Turn: player_1

  0 1 2 3 4 5
0 . b . b . b
1 b . . . b .
2 . b . . . .
3 . . . . r .
4 . r . . . r
5 r . r . r .
Turn: player_0

...

  0 1 2 3 4 5
0 . . . . . .
1 . . . . . .
2 . r . . . r
3 . . . . r .
4 . . . . . .
5 r . . . R .
Game over
Trained episode finished in 35 steps
Final rewards: {'player_0': 1.0, 'player_1': -1.0}
```

## 3) Final Cumulative Reward Summary

| Metric | Value |
|---|---|
| Total episodes | 1000 |
| Total steps | 41706 |
| Cumulative final reward (player_0) | +743.000 |
| Cumulative final reward (player_1) | -743.000 |
| Epsilon schedule | start 0.200, end 0.020, decay 0.995, final 0.020 |
| Win counts | player_0: 849, player_1: 106, draw: 45 |

## 4) Brief Explanation of Agent Logic and Function Approximation Design Choices

The agent is a self-play Actor-Critic model for 6x6 checkers with one shared policy for both players.

Key design choices:

- Player-centered observation transform:
  - For player_1 turns, board perspective is normalized (flip/sign convention) so one model can learn consistently from either side.
- Linear function approximation:
  - Actor: linear policy head with masked softmax so illegal moves get zero probability.
  - Critic: linear state-value estimator.
- Learning signal:
  - Critic uses TD(0).
  - Actor uses policy-gradient updates weighted by TD advantage.
- Self-play strategy:
  - Trains against current and older policy snapshots to reduce overfitting to a fixed opponent.
- Evaluation stability:
  - Uses decaying epsilon-greedy during trained evaluation to reduce deterministic trajectories and improve robustness.

Reference source in project: brief_explanation.md
