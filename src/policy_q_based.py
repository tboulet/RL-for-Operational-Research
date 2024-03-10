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
from src.constants import EPSILON
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


class PolicyGreedy(PolicyQBased):
    """The greedy policy for Q-learning. It always picks the action with the highest Q-value."""

    def get_action_and_prob(
        self, state: Any, available_actions: List[Any], is_eval: bool = False
    ) -> Action:
        return max(available_actions, key=lambda a: self.q_values[state][a]), 1


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
        greedy_action = max(available_actions, key=lambda a: self.q_values[state][a])
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


class PolicyBoltzmann(PolicyQBased):

    def __init__(self, q_values: QValues, temperature: Union[float, int, Scheduler]):
        """The Boltzmann policy for Q-learning. It assigns probabilities to each action proportional to the exponentiated Q-value.
        The temperature parameter controls the randomness of the policy. A high temperature leads to a more random policy, while a low temperature leads to a more deterministic policy.

        Args:
            q_values (QValues): the Q values of the policy
            temperature (Union[float, int, Scheduler]): the temperature value or scheduler config.
        """
        super().__init__(q_values=q_values)
        self.temperature: Scheduler = get_scheduler(config_or_value=temperature)

    def get_action_and_prob(
        self, state: Any, available_actions: List[Any], is_eval: bool = False
    ) -> Action:
        # In eval mode, act greedily
        if is_eval:
            return max(available_actions, key=lambda a: self.q_values[state][a]), 1

        # Get the temperature
        temperature = self.temperature.get_value()

        # Compute the probabilities
        q_values = np.array([self.q_values[state][a] for a in available_actions])
        exp_q_values = np.exp(q_values / temperature)
        probs = exp_q_values / np.sum(exp_q_values)

        # Choose the action
        action = np.random.choice(available_actions, p=probs)

        # Update the temperature scheduler
        self.temperature.increment_step()

        return action, probs[available_actions.index(action)]


class PolicyUCB(PolicyQBased):

    def __init__(self, q_values: QValues, ucb_constant: Union[float, int, Scheduler]):
        """The Upper Confidence Bound (UCB) policy for Q-learning.
        It pick the action that maximizes a trade-off between exploitation and exploration.
        The exploitation term is the Q-value, while the exploration term is a term that explodes when the action has not been tried often in that state.

        Args:
            q_values (QValues): the Q values of the policy
            ucb_constant (Union[float, int, Scheduler]): the exploration bonus value or scheduler config.
        """
        super().__init__(q_values=q_values)
        self.ucb_constant: Scheduler = get_scheduler(config_or_value=ucb_constant)
        self.n_seen_observed = defaultdict(lambda: defaultdict(int))

    def get_action_and_prob(
        self, state: Any, available_actions: List[Any], is_eval: bool = False
    ) -> Action:
        # In eval mode, act greedily
        if is_eval:
            return max(available_actions, key=lambda a: self.q_values[state][a]), 1

        # Get the exploration bonus
        constant_ucb = self.ucb_constant.get_value()

        # Compute the UCB
        n_s = sum(self.n_seen_observed[state][a] for a in available_actions)
        ucb_values = [
            self.q_values[state][a]
            + constant_ucb
            * np.sqrt(
                np.log(n_s + 1) / (self.n_seen_observed[state][a] + EPSILON)
            )
            for a in available_actions
        ]

        # Choose the action
        action = max(
            available_actions, key=lambda a: ucb_values[available_actions.index(a)]
        )

        # Update the exploration bonus scheduler
        self.ucb_constant.increment_step()

        # Update the number of times the action has been seen
        self.n_seen_observed[state][action] += 1

        return action, 1
