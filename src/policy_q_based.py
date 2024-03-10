# Logging
from collections import defaultdict
import wandb
from tensorboardX import SummaryWriter

# Config system
import hydra
from omegaconf import OmegaConf, DictConfig

# Utils
from tqdm import tqdm
import datetime
from time import time
from typing import Dict, List, Optional, Type, Any, Tuple, Union
import cProfile

# ML libraries
import random
import numpy as np

# File specific
from abc import ABC, abstractmethod
from src.schedulers import Scheduler, get_scheduler

from src.typing import Action, QValues


class PolicyQBased(ABC):
    """The base interface for all Q-based policies. This class
    should be subclassed when implementing a new policy based on Q values.

    This class requires to implement the "get_action_and_prob" method.
    """

    def __init__(self, q_values: QValues):
        """Initialize the policy with the given configuration.

        Args:
            q_values (QValues): the Q values of the policy
        """
        self.q_values = q_values

    @abstractmethod
    def get_action_and_prob(
        self, state: Any, available_actions: List[Any], is_eval: bool = False
    ) -> Tuple[Action, Optional[float]]:
        """Perform an action based on the state. Also return the probability of the action being chosen if possible, or None if not.

        Args:
            state (Any): the current state (or observation)
            available_actions (List[Any]): the list of available actions for the agent to choose from
            is_eval (bool, optional): whether the agent is evaluating or not. Defaults to False.

        Returns:
            Action: the action to perform according to the agent
            Optional[float]: the probability of the action being chosen, or None if not available
        """


class PolicyEpsilonGreedy(PolicyQBased):

    def __init__(self, q_values: QValues, epsilon: Union[float, int, Scheduler]):
        """The epsilon-greedy policy for Q-learning.

        Args:
            q_values (QValues): the Q values of the policy
            epsilon (Union[float, int, Scheduler]): the epsilon value or scheduler config.
        """
        super().__init__(q_values=q_values)
        self.epsilon: Scheduler = get_scheduler(config_or_value=epsilon)

    def get_action_and_prob(
        self, state: Any, available_actions: List[Any], is_eval: bool = False
    ) -> Action:
        greedy_action = max(
                available_actions, key=lambda a: self.q_values[state][a]
            )
        if is_eval:
            # In eval mode, this is the greedy policy
            return greedy_action, 1

        else:
            # In training mode, we use epsilon-greedy
            n_actions = len(available_actions)
            eps = self.epsilon.get_value()

            # Pick action
            if np.random.uniform() < eps:
                # With probability epsilon, we explore
                eps = self.epsilon.get_value()
                action = random.choice(available_actions)
            else:
                action = greedy_action

            # Compute prob
            if action == greedy_action:
                prob = 1 - eps + eps / n_actions
            else:
                prob = eps / n_actions

            # Update the epsilon scheduler
            self.epsilon.increment_step()
            
            return action, prob
