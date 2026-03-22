# Documentation
- [https://pettingzoo.farama.org/tutorials/custom_environment/](https://pettingzoo.farama.org/tutorials/custom_environment/)


# Agent Environment Cycle (AEC) vs. Parallel Environments
- By default, PettingZoo models games as [Agent Environment Cycle (AEC) environments](https://pettingzoo.farama.org/api/aec/#about-aec)
- In an AEC environment, agents act sequentially, receiving updated observations and rewards before taking an action
  - ```AgentSelector``` steps through agents in a cycle
- We have a secondary [parallel API](https://pettingzoo.farama.org/api/parallel/) for environments where all agents have simultaneous actions and observations
- Examples can be found on [the Environment Creation page](https://pettingzoo.farama.org/content/environment_creation/)

# Skeleton code

```python
from pettingzoo import ParallelEnv

class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self):
        pass

    def reset(self, seed=None, options=None):
        pass

    def step(self, actions):
        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
```

# @functools.lru_cache
Decorator to wrap a function with a memoizing callable that saves up to the maxsize most recent calls. It can save time when an expensive or I/O bound function is periodically called with the same arguments. ([Reference](https://docs.python.org/3/library/functools.html#functools.lru_cache))