from typing import Dict, Type
from environments.base_environment import BaseOREnvironment


def get_sparsified_env_class(EnvClass: Type[BaseOREnvironment]) -> Type[BaseOREnvironment]:
    """Sparsify the rewards of an environment by accumulating the rewards and returning the total reward at the end of the episode.
    Pros: Remove the bias towards near-term rewards
    Cons: The agent may not be able to learn from the rewards as effectively
    
    Args:
        EnvClass (Type[BaseOREnvironment]): the class of the environment to sparsify

    Returns:
        Type[BaseOREnvironment]: the class of the sparsified environment
    """
    
    class EnvSparsified(EnvClass):

        def __init__(self, config: Dict):
            self.total_reward_sparsified = None
            super().__init__(config)

        def reset(self, *args, **kwargs):
            self.total_reward_sparsified = 0
            return super().reset(*args, **kwargs)

        def step(self, action, *args, **kwargs):
            assert (
                self.total_reward_sparsified is not None
            ), "You must call `reset` before calling `step`"
            obs, reward, is_trunc, done, info = super().step(action, *args, **kwargs)
            self.total_reward_sparsified += reward
            if done:
                return obs, self.total_reward_sparsified, is_trunc, done, info
            else:
                return obs, 0, is_trunc, done, info

    return EnvSparsified


