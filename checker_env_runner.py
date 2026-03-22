import numpy as np
import sys
from contextlib import redirect_stderr, redirect_stdout

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
# Uses a random legal-action policy - NOT AN AGENT
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


if __name__ == "__main__":
    log_path = "checker_runner_output.log"
    with open(log_path, "w", encoding="utf-8") as log_file:
        tee_out = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"Logging terminal output to {log_path}")
            run_episode()
