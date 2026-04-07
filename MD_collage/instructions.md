# Project Run Instructions

This guide explains how a new user can run the custom 6x6 Checkers environment and the Actor-Critic self-play agent.

## 1) Prerequisites

- Python 3.10 or newer
- pip
- Terminal in the project root folder

## 2) Install dependencies

Use one of the following methods.

Option A: with Makefile

1. make setup

Option B: direct pip install

1. python -m pip install -r requirements.txt

If your virtual environment python is in a different path, run the same install command with that python executable.

## 3) Verify Task 1 environment behavior

This runs random legal actions and prints board progression.

1. python checker_env_runner.py 

Expected output:

- Repeated board states in terminal
- Turn lines for player_0 and player_1
- Game over line
- Total rewards line

The output is also saved to checker_runner_output.log.

## 4) Train Actor-Critic self-play agent (Task 2)

Important: retrain the model before generating final trained-run results.
If you changed agent logic, runner behavior, or evaluation settings, generate fresh weights.

Quick smoke run:

1. python myagent.py --episodes 5 --eval_every 5 --snapshot_interval 2 --save_path ac_smoke.npz

Fuller training run:

1. python myagent.py --episodes 3000 --eval_every 200 --snapshot_interval 200 --save_path ac_selfplay_weights.npz

Expected output during/after training:

- Rolling reward lines every eval interval
- Model saved message
- Evaluation mean return summary

## 5) Run trained sample games with board progression

This satisfies the sample trained run requirement and logs cumulative rewards across episodes.

1. python myrunner.py --mode trained --model_path ac_selfplay_weights.npz --episodes 1000 --max_steps 300 --render_mode human --log_path trained_sample_run.log

Use ac_selfplay_weights.npz if you ran full training.

Note: trained mode now uses built-in decaying epsilon-greedy behavior across episodes.
This helps avoid identical trajectories and provides more realistic performance summaries.

Expected output:

- Per-episode board progression
- Per-episode summary:
	- Trained episode finished in X steps
	- Final rewards for both players
- Final summary:
	- Total episodes
	- Total steps
	- Cumulative final reward for each player
	- Win counts

## 6) Optional fast summary-only mode

This disables board rendering and prints only summaries.

1. python myrunner.py --mode trained --model_path ac_selfplay_weights.npz --episodes 50 --render_mode none --log_path trained_summary_only.log

## 7) Common troubleshooting

- ModuleNotFoundError for gymnasium or pettingzoo:
	reinstall requirements.txt with the same python executable used to run scripts.
- Package/version mismatch:
	use the exact required versions in requirements.txt (torch==2.8.0, tensorflow==2.19.0, scikit-learn==1.8.0).
- Command not found for a virtual environment python path:
	use whichever python executable exists in your environment and run the same script commands.
