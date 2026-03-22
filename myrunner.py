# The Checkers env
import argparse
import numpy as np
import sys
from contextlib import redirect_stderr, redirect_stdout

from myagent import ACConfig, ActorCriticSelfPlay
from mycheckersenv import CustomEnvironment


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

# Run a single episode of the environment with random actions, printing out the final rewards at the end.
# Provides a simple display of the environment (gameboard) at each step.
def run_episode(seed=42, render_mode="human", max_steps=300):
    env = CustomEnvironment(render_mode=render_mode)
    observations, infos = env.reset(seed=seed)

    step_count = 0
    last_rewards = {agent: 0.0 for agent in env.possible_agents}

    while env.agents and step_count < max_steps:
        agent = env.current_agent
        mask = observations[agent]["action_mask"]
        legal_actions = np.flatnonzero(mask)

        if legal_actions.size == 0:
            # No legal actions should normally correspond to terminal logic in env.
            actions = {agent: 0}
        else:
            action = int(np.random.choice(legal_actions))
            actions = {agent: action}

        observations, rewards, terminations, truncations, infos = env.step(actions)
        last_rewards = rewards
        step_count += 1

        if any(terminations.values()) or any(truncations.values()):
            break

    # The winning player will have a reward of 1, the losing player will have a reward of -1,
    # if the game is truncated due to max_steps, both players will have a reward of 0.
    print(f"Episode finished in {step_count} steps")
    print(f"Final rewards: {last_rewards}")
    return step_count, last_rewards


def run_trained_episode(policy, seed=42, render_mode="human", max_steps=300, epsilon=0.0):

    env = CustomEnvironment(render_mode=render_mode)
    observations, infos = env.reset(seed=seed)

    step_count = 0
    last_rewards = {agent: 0.0 for agent in env.possible_agents}

    while env.agents and step_count < max_steps:
        agent = env.current_agent
        action = policy.act(observations[agent], agent, epsilon=epsilon)
        observations, rewards, terminations, truncations, infos = env.step({agent: action})
        last_rewards = rewards
        step_count += 1

        if any(terminations.values()) or any(truncations.values()):
            break

    print(f"Trained episode finished in {step_count} steps")
    print(f"Final rewards: {last_rewards}")
    return step_count, last_rewards


def run_many_episodes(mode, episodes, seed, render_mode, max_steps, model_path):
    cumulative_rewards = {"player_0": 0.0, "player_1": 0.0}
    cumulative_steps = 0
    wins = {"player_0": 0, "player_1": 0, "draw": 0}

    policy = None
    epsilon_start = 0.20
    epsilon_end = 0.02
    epsilon_decay = 0.995
    if mode == "trained":
        cfg = ACConfig(episodes=0, seed=seed)
        policy = ActorCriticSelfPlay(CustomEnvironment, cfg)
        policy.load(model_path)

    for ep in range(1, episodes + 1):
        print("=" * 60)
        print(f"Episode {ep}/{episodes}")

        ep_seed = seed + ep - 1
        if mode == "trained":
            episode_epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** (ep - 1)))
            print(f"Epsilon: {episode_epsilon:.3f}")
            step_count, rewards = run_trained_episode(
                policy=policy,
                seed=ep_seed,
                render_mode=render_mode,
                max_steps=max_steps,
                epsilon=episode_epsilon,
            )
        else:
            step_count, rewards = run_episode(
                seed=ep_seed,
                render_mode=render_mode,
                max_steps=max_steps,
            )

        cumulative_steps += step_count
        for agent in cumulative_rewards:
            cumulative_rewards[agent] += float(rewards.get(agent, 0.0))

        p0_reward = float(rewards.get("player_0", 0.0))
        p1_reward = float(rewards.get("player_1", 0.0))
        if p0_reward > p1_reward:
            wins["player_0"] += 1
        elif p1_reward > p0_reward:
            wins["player_1"] += 1
        else:
            wins["draw"] += 1

    print("=" * 60)
    print("Final Summary")
    print(f"Total episodes: {episodes}")
    print(f"Total steps: {cumulative_steps}")
    print(f"Cumulative final reward - player_0: {cumulative_rewards['player_0']:+.3f}")
    print(f"Cumulative final reward - player_1: {cumulative_rewards['player_1']:+.3f}")
    if mode == "trained":
        final_eps = max(epsilon_end, epsilon_start * (epsilon_decay ** (episodes - 1)))
        print(
            f"Epsilon schedule - start: {epsilon_start:.3f}, "
            f"end: {epsilon_end:.3f}, decay: {epsilon_decay:.3f}, "
            f"final episode epsilon: {final_eps:.3f}"
        )
    print(
        "Win counts - "
        f"player_0: {wins['player_0']}, "
        f"player_1: {wins['player_1']}, "
        f"draw: {wins['draw']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run checkers env with random or trained policy")
    parser.add_argument("--mode", choices=["random", "trained"], default="trained")
    parser.add_argument("--model_path", type=str, default="ac_selfplay_weights.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render_mode", choices=["human", "none"], default="human")
    parser.add_argument("--log_path", type=str, default="checker_runner_output.log")
    args = parser.parse_args()

    log_path = args.log_path
    with open(log_path, "w", encoding="utf-8") as log_file:
        tee_out = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"Logging terminal output to {log_path}")
            run_many_episodes(
                mode=args.mode,
                episodes=args.episodes,
                seed=args.seed,
                render_mode=None if args.render_mode == "none" else "human",
                max_steps=args.max_steps,
                model_path=args.model_path,
            )
