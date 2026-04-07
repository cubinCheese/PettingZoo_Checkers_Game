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
A key issue was repeated deterministic trajectories -- a major challenge faced in our prior project.
- Trained evaluation uses decaying epsilon-greedy behavior across episodes.

Metrics are reported in submission_report.md and run logs.