# 6x6 Checkers: Self-Play Actor-Critic

## Overview
This project implements a self-play Actor-Critic agent for 6x6 Checkers.

- One shared policy controls both players.
- Observations are converted into a player-centered view (board flip and sign adjustment for player_1).
- This lets one model learn from both sides consistently.

## Function Approximation Design
The model uses linear function approximation for simplicity and stability.

- Actor: linear policy head with masked softmax over legal moves only (no illegal move probability).
- Critic: linear state-value estimator.
- Training signal: temporal-difference (TD) error.

Update logic:

- Critic uses TD(0) updates.
- Actor uses policy-gradient updates weighted by the TD advantage signal.

## Self-Play Strategy
- The learner trains against current and older policy snapshots.
- This avoids overfitting to a single fixed opponent.

## Reducing Deterministic Behavior
A key issue was repeated deterministic trajectories -- a major challenge I faced in our prior project.
Drawing from that experience...
- Trained evaluation now uses default decaying epsilon-greedy behavior across episodes.
- This improves trajectory diversity and yields more realistic aggregate performance estimates.

## Final Summary
| Metric | Value |
|---|---|
| Total episodes | 1000 |
| Total steps | 41706 |
| Cumulative final reward (player_0) | +743.000 |
| Cumulative final reward (player_1) | -743.000 |
| Epsilon schedule | start 0.200, end 0.020, decay 0.995, final 0.020 |
| Win counts | player_0: 849, player_1: 106, draw: 45 |