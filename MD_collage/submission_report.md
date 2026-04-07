# CS272 PA2 - Custom Checkers Environment Report (AEC Refactor)

## 1) Public GitHub Repository

Project repository (clickable):

[https://github.com/cubinCheese/PettingZoo_Checkers_Game](https://github.com/cubinCheese/PettingZoo_Checkers_Game)

## 2) Sample Run with Trained Agents (Board State Progression)

Source: test_results/trained_sample_run.log

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
| Total steps | 46403 |
| Cumulative final reward (player_0) | +727.000 |
| Cumulative final reward (player_1) | -727.000 |
| Epsilon schedule | start 0.200, end 0.020, decay 0.995, final 0.020 |
| Win counts | player_0: 847, player_1: 120, draw: 33 |

These values should be regenerated if training or evaluation settings change.

## 4) Brief Explanation of Agent Logic and Function Approximation Design Choices

The project uses a self-play Actor-Critic agent for 6x6 checkers with one shared policy for both players. Observations are converted to a player-centered view by flipping and sign-adjusting the board for player_1, so the same model can learn both sides consistently. The actor is a linear policy head with masked softmax over legal moves only, and the critic is a linear state-value estimator. Learning uses TD(0) updates for the critic and policy-gradient updates weighted by TD error for the actor. Training uses snapshot self-play: the learner competes against frozen historical versions of itself, which reduces overfitting to a single opponent. Trained evaluation applies decaying epsilon-greedy exploration to avoid repeated deterministic trajectories and produce more realistic aggregate results.

Reference source in project: brief_explanation.md
