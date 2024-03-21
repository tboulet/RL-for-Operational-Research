from typing import Dict, Type
from environments.base_environment import BaseOREnvironment


def get_normalized_reward_env_class(
    EnvClass: Type[BaseOREnvironment],
) -> Type[BaseOREnvironment]:
    """Normalize the rewards of an environment by dividing the reward by the reward range at the end of the episode.

    Args:
        EnvClass (Type[BaseOREnvironment]): the class of the environment to normalize the rewards of

    Returns:
        Type[BaseOREnvironment]: the class of the environment with normalized rewards
    """

    class EnvNormalized(EnvClass):

        def __init__(self, config: Dict):
            self.reward_range_delta = None
            super().__init__(config)

        def reset(self, *args, **kwargs):
            # Compute the reward range at the beginning of the episode
            worst_reward = super().get_worst_reward()
            best_reward = super().get_optimal_reward()
            if worst_reward is not None and best_reward is not None:
                self.reward_range_delta = best_reward - worst_reward
                if self.reward_range_delta < 0:
                    # Check if the rewards are not inverted
                    raise ValueError(
                        f"The optimal reward {best_reward} is smaller than the worst reward {worst_reward}"
                    )
                elif self.reward_range_delta == 0:
                    # If the rewards are constant, we set the range to 1 (no normalization)
                    self.reward_range_delta = 1
            else:
                # If the optimal or worst reward is not defined, we set the range to 1 (no normalization)
                self.reward_range_delta = 1

            return super().reset(*args, **kwargs)

        def step(self, action, *args, **kwargs):
            assert (
                self.reward_range_delta is not None
            ), "You must call `reset` before calling `step`"
            obs, reward, is_trunc, done, info = super().step(action, *args, **kwargs)
            return obs, reward / self.reward_range_delta, is_trunc, done, info

        def get_optimal_reward(self):
            optimal_reward = super().get_optimal_reward()
            if optimal_reward is not None:
                return optimal_reward / self.reward_range_delta
            else:
                return None

        def get_worst_reward(self):
            worst_reward = super().get_worst_reward()
            if worst_reward is not None:
                return worst_reward / self.reward_range_delta
            else:
                return None

    return EnvNormalized
