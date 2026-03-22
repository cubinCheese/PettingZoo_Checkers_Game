# The AC agent
import argparse
from dataclasses import dataclass

import numpy as np

from mycheckersenv import CustomEnvironment


@dataclass
class ACConfig:
	# Hyperparameters for the Actor-Critic agent and training loop.
	gamma: float = 0.99
	actor_lr: float = 0.01
	critic_lr: float = 0.05
	entropy_coef: float = 0.001
	episodes: int = 3000
	eval_every: int = 200
	snapshot_interval: int = 200
	snapshot_pool_size: int = 8
	max_steps_per_episode: int = 400
	seed: int = 42

# The AC agent is implemented in the ActorCriticSelfPlay class, which includes methods for training, acting, evaluating, saving, and loading the model. 
# The agent uses a simple linear function approximation for both the actor and critic, and trains using a policy gradient method with a TD error as the advantage. 
# The agent also maintains a pool of opponent snapshots for self-play training.
class ActorCriticSelfPlay:
	def __init__(self, env_cls, config: ACConfig):
		self.env_cls = env_cls
		self.cfg = config
		self.rng = np.random.default_rng(config.seed)

		probe_env = self.env_cls(render_mode=None)
		obs, _ = probe_env.reset(seed=config.seed)
		first_agent = probe_env.current_agent
		self.num_actions = int(probe_env.action_space(first_agent).n)
		self.state_dim = self._state_features(obs[first_agent], first_agent).shape[0]

		# Actor logits: for action a, score(a|s) = W[a] dot s + b[a]
		self.actor_w = self.rng.normal(0.0, 0.01, size=(self.num_actions, self.state_dim))
		self.actor_b = np.zeros(self.num_actions, dtype=np.float64)

		# Critic: V(s) = v dot s + c
		self.critic_v = np.zeros(self.state_dim, dtype=np.float64)
		self.critic_c = 0.0

		self.snapshots = [self._snapshot()]
    
    # The _snapshot method captures the current parameters of the actor.
	def _snapshot(self):
		return {
			"actor_w": self.actor_w.copy(),
			"actor_b": self.actor_b.copy(),
		}

    # The _sample_opponent_snapshot method randomly samples a snapshot from the pool of opponent snapshots for self-play.
	def _sample_opponent_snapshot(self):
		return self.snapshots[int(self.rng.integers(0, len(self.snapshots)))]

    # The _state_features method processes the raw observation from the environment into a feature vector for the neural network. 
	def _state_features(self, observation, agent):
		board = observation["board"].astype(np.float64)

		# Canonicalize perspective so one policy can be shared by both players.
		if agent == "player_1":
			board = -np.flipud(np.fliplr(board))

		own_men = np.sum(board == 1)
		own_kings = np.sum(board == 2)
		opp_men = np.sum(board == -1)
		opp_kings = np.sum(board == -2)

		# Compact global features + full board encoding.
		scalars = np.array(
			[
				own_men,
				own_kings,
				opp_men,
				opp_kings,
				own_men + 1.5 * own_kings - (opp_men + 1.5 * opp_kings),
				1.0,
			],
			dtype=np.float64,
		)

		return np.concatenate([board.flatten(), scalars], axis=0)

    # The _masked_softmax method computes a softmax over the action logits, masking out illegal actions by assigning them a large negative value before exponentiating.
	def _masked_softmax(self, logits, legal_actions):
		masked = np.full_like(logits, -1e9, dtype=np.float64)   # Mask illegal actions with large negative value.
		masked[legal_actions] = logits[legal_actions]           # Assign logits to legal actions.
		max_logit = np.max(masked[legal_actions])               # For numerical stability, subtract max logit before exponentiating.
		exps = np.zeros_like(logits, dtype=np.float64)          # Compute softmax.
		exps[legal_actions] = np.exp(masked[legal_actions] - max_logit)  # Only exponentiate legal actions.
		z = np.sum(exps[legal_actions])                         # Normalization constant (sum of exponentials of legal actions).
		probs = np.zeros_like(logits, dtype=np.float64)         # Final probability distribution over actions, with zeros for illegal actions.
		probs[legal_actions] = exps[legal_actions] / z          # Normalize to get probabilities.
		return probs

    # Computes the action probabilities for a given state and legal actions using the actor's parameters.
	def _policy_probs(self, features, legal_actions, actor_w, actor_b):
		logits = actor_w @ features + actor_b
		return self._masked_softmax(logits, legal_actions)

    # Samples an action from the given probability distribution over legal actions.
	def _sample_action(self, probs, legal_actions):
		p = probs[legal_actions]
		idx = int(self.rng.choice(len(legal_actions), p=p))
		return int(legal_actions[idx])

    # Computes the value of a state using the critic's parameters.
	def _value(self, features):
		return float(np.dot(self.critic_v, features) + self.critic_c)

    # Performs a single update step for both the actor and critic based on the observed transition (s, a, r, s_next) and the legal actions in state s.
	def _update(self, s, a, r, s_next, done, legal_actions_s):
		v_s = self._value(s)
		v_next = 0.0 if done else self._value(s_next)
		td_error = r + self.cfg.gamma * v_next - v_s

		# Critic update (semi-gradient TD(0))
		self.critic_v += self.cfg.critic_lr * td_error * s
		self.critic_c += self.cfg.critic_lr * td_error

		# Actor update: policy gradient with TD error as advantage.
		probs = self._policy_probs(s, legal_actions_s, self.actor_w, self.actor_b)
		grad_logits = -probs
		grad_logits[a] += 1.0

		self.actor_w += self.cfg.actor_lr * td_error * np.outer(grad_logits, s)
		self.actor_b += self.cfg.actor_lr * td_error * grad_logits

		# Small entropy bonus to keep exploration alive.
		legal_probs = probs[legal_actions_s]
		entropy_grad = np.zeros_like(probs)
		entropy_grad[legal_actions_s] = -(np.log(np.clip(legal_probs, 1e-12, 1.0)) + 1.0)
		self.actor_b += self.cfg.actor_lr * self.cfg.entropy_coef * entropy_grad

		return td_error

    # The train method runs the main training loop for the specified number of episodes, performing self-play and periodically evaluating the agent against random opponents.
	def train(self):
		rolling = []

        # Main training loop
		for ep in range(1, self.cfg.episodes + 1):
			env = self.env_cls(render_mode=None, max_moves=self.cfg.max_steps_per_episode)
			observations, _ = env.reset(seed=int(self.rng.integers(0, 10**9)))

			learner_agent = "player_0" if self.rng.random() < 0.5 else "player_1"
			opponent_agent = "player_1" if learner_agent == "player_0" else "player_0"
			opponent_snapshot = self._sample_opponent_snapshot()

			ep_reward = 0.0
			steps = 0
            
            # Main game loop
			while env.agents and steps < self.cfg.max_steps_per_episode:
				agent = env.current_agent
				obs = observations[agent]
				legal_actions = np.flatnonzero(obs["action_mask"])

				if legal_actions.size == 0:
					action = 0
				else:
					features = self._state_features(obs, agent)
					if agent == learner_agent:
						probs = self._policy_probs(features, legal_actions, self.actor_w, self.actor_b)
						action = self._sample_action(probs, legal_actions)
					else:
						probs = self._policy_probs(
							features,
							legal_actions,
							opponent_snapshot["actor_w"],
							opponent_snapshot["actor_b"],
						)
						action = self._sample_action(probs, legal_actions)

				next_observations, rewards, terminations, truncations, _ = env.step({agent: int(action)})

				done = any(terminations.values()) or any(truncations.values())

                # If the acting agent is the learner, perform an update step using the observed transition.
				if agent == learner_agent:
					s = self._state_features(obs, agent)
					if not done and env.agents:
						s_next_obs = next_observations[learner_agent]
						s_next = self._state_features(s_next_obs, learner_agent)
					else:
						s_next = np.zeros_like(s)

					self._update(
						s=s,
						a=int(action),
						r=float(rewards.get(learner_agent, 0.0)),
						s_next=s_next,
						done=done,
						legal_actions_s=legal_actions,
					)

                # Accumulate reward for the learner agent and move to the next step.
				ep_reward += float(rewards.get(learner_agent, 0.0))
				observations = next_observations
				steps += 1

				if done:
					break
            
            # At the end of the episode, record the reward in the rolling window and periodically save snapshots and print progress.
			rolling.append(ep_reward)
			if len(rolling) > 100:
				rolling.pop(0)

            # Periodically save a snapshot of the current policy to the pool of opponent snapshots for future self-play.
			if ep % self.cfg.snapshot_interval == 0:
				self.snapshots.append(self._snapshot())
				if len(self.snapshots) > self.cfg.snapshot_pool_size:
					self.snapshots.pop(0)

            # Periodically evaluate the current policy against random opponents and print out the average reward over the rolling window of recent episodes.
			if ep % self.cfg.eval_every == 0:
				avg = float(np.mean(rolling)) if rolling else 0.0
				print(f"Episode {ep:5d} | rolling learner reward(100): {avg:+.3f} | snapshots: {len(self.snapshots)}")

    # The act method takes an observation and legal actions for a given agent, and returns an action sampled from the current policy's probability distribution over legal actions, with epsilon-greedy exploration.
	def act(self, observation, agent, epsilon=0.0):
		legal_actions = np.flatnonzero(observation["action_mask"])
		if legal_actions.size == 0:
			return 0
		if self.rng.random() < float(epsilon):
			return int(legal_actions[int(self.rng.integers(0, len(legal_actions)))])
		s = self._state_features(observation, agent)
		probs = self._policy_probs(s, legal_actions, self.actor_w, self.actor_b)
		return int(legal_actions[int(np.argmax(probs[legal_actions]))])

    # The evaluate method runs a specified number of episodes against random opponents and returns the average reward for the learner agent.
	def evaluate(self, episodes=50):
		returns = []
		for _ in range(episodes):
			env = self.env_cls(render_mode=None, max_moves=self.cfg.max_steps_per_episode)
			observations, _ = env.reset(seed=int(self.rng.integers(0, 10**9)))
			learner_agent = "player_0"
			total = 0.0

			while env.agents:
				agent = env.current_agent
				if agent == learner_agent:
					action = self.act(observations[agent], agent)
				else:
					legal = np.flatnonzero(observations[agent]["action_mask"])
					action = int(legal[int(self.rng.integers(0, len(legal)))]) if legal.size else 0

				observations, rewards, terminations, truncations, _ = env.step({agent: action})
				total += float(rewards.get(learner_agent, 0.0))
				if any(terminations.values()) or any(truncations.values()):
					break

			returns.append(total)

		mean_ret = float(np.mean(returns)) if returns else 0.0
		print(f"Evaluation over {episodes} episodes, mean return: {mean_ret:+.3f}")
		return mean_ret

    # The save method saves the current parameters of the actor and critic to a file, and the load method loads parameters from a file.
	def save(self, path="ac_selfplay_weights.npz"):
		np.savez(
			path,
			actor_w=self.actor_w,
			actor_b=self.actor_b,
			critic_v=self.critic_v,
			critic_c=np.array([self.critic_c], dtype=np.float64),
		)
		print(f"Saved model to {path}")

    # The load method loads the actor and critic parameters from a file, allowing for resuming training or evaluation from a saved model.
	def load(self, path="ac_selfplay_weights.npz"):
		data = np.load(path)
		self.actor_w = data["actor_w"]
		self.actor_b = data["actor_b"]
		self.critic_v = data["critic_v"]
		self.critic_c = float(data["critic_c"][0])
		print(f"Loaded model from {path}")


def main():
	parser = argparse.ArgumentParser(description="Actor-Critic self-play for 6x6 Checkers")
	parser.add_argument("--episodes", type=int, default=3000)
	parser.add_argument("--eval_every", type=int, default=200)
	parser.add_argument("--snapshot_interval", type=int, default=200)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--save_path", type=str, default="ac_selfplay_weights.npz")
	parser.add_argument("--evaluate_only", action="store_true")
	parser.add_argument("--load_path", type=str, default="")
	args = parser.parse_args()

	cfg = ACConfig(
		episodes=args.episodes,
		eval_every=args.eval_every,
		snapshot_interval=args.snapshot_interval,
		seed=args.seed,
	)
	trainer = ActorCriticSelfPlay(CustomEnvironment, cfg)

	if args.load_path:
		trainer.load(args.load_path)

	if not args.evaluate_only:
		trainer.train()
		trainer.save(args.save_path)

	trainer.evaluate(episodes=50)


if __name__ == "__main__":
	main()