.PHONY: setup env-check train-smoke train-full run-trained run-trained-fast eval-only

setup:
	python3.10 -m pip install -r requirements.txt

env-check:
	python3.10 checker_env_runner.py

train-smoke:
	python3.10 myagent.py --episodes 5 --eval_every 5 --snapshot_interval 2 --save_path ac_smoke.npz

train-full:
	python3.10 myagent.py --episodes 3000 --eval_every 200 --snapshot_interval 200 --save_path ac_selfplay_weights.npz

run-trained:
	python3.10 myrunner.py --mode trained --model_path ac_selfplay_weights.npz --episodes 1000 --max_steps 300 --render_mode human --log_path trained_sample_run.log

run-trained-fast:
	python3.10 myrunner.py --mode trained --model_path ac_selfplay_weights.npz --episodes 50 --max_steps 300 --render_mode none --log_path trained_summary_only.log

eval-only:
	python3.10 myagent.py --evaluate_only --load_path ac_selfplay_weights.npz