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
from typing import Dict, List, Optional, Type, Any, Tuple
import cProfile

# ML libraries
import random
import numpy as np

# File specific
from abc import ABC, abstractmethod
from src.constants import INF
from src.initialization import initialize_tabular_q_values
from src.learners.base_learner import BaseLearner
from src.metrics import get_q_values_metrics, get_scheduler_metrics_of_object
from src.policies.policy_q_based import (
    PolicyBoltzmann,
    PolicyEpsilonGreedy,
    PolicyGreedy,
    PolicyQBased,
    PolicyUCB,
)
from src.schedulers import Scheduler

# Project imports
from src.typing import State, Action
from src.utils import try_get


class BaseRLAlgorithm(ABC):
    """The base interface for all reinforcement learning algorithms. This class
    should be subclassed when implementing a new RL algorithm.

    It requires to implement the `act` and `update` methods.
    """

    def __init__(self, config: Dict):
        """Initialize the agent with the given configuration.

        Args:
            config (Dict): the configuration of the agent
        """
        self.config = config

    @abstractmethod
    def act(
        self, state: State, available_actions: List[Action], is_eval: bool = False
    ) -> Action:
        """Perform an action based on the state. This is the inference method of the agent.

        Args:
            state (State): the current state (or observation)
            available_actions (List[Action]): the list of available actions for the agent to choose from
            is_eval (bool, optional): whether the agent is evaluating or not. Defaults to False.

        Returns:
            Action: the action to perform according to the agent
        """

    @abstractmethod
    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool,
    ) -> Optional[Dict[str, float]]:
        """Learn from the transition `(state, action, reward, next_state)`.
        This is the method that updates the agent's knowledge.

        Args:
            state (State): the current state
            action (Action): the action performed
            reward (float): the reward obtained after performing the action
            next_state (State): the next state the algorithm transitions to after performing the action
            done (bool): whether the episode is over or not

        Returns:
            Optional[Dict[str, float]]: the metrics of the update step. This is useful for logging purposes, or None if no metrics are available.
        """

    # ============ Helper methods ============

    def initialize_q_values(self, config: Dict) -> Dict[State, Dict[Action, float]]:
        """Initialize the Q-values of the agent based on the configuration.

        Args:
            config (Dict): the configuration of the agent

        Returns:
            Dict[State, Dict[Action, float]]: the initialized Q-values of the agent
        """
        # Get the method of Q-value initialization
        assert "method_q_value_initialization" in config, (
            "The method of Q-value initialization is not specified in the configuration. Please specify it using the "
            "key 'method_q_value_initialization' in the configuration file."
        )
        return initialize_tabular_q_values(
            method_q_value_initialization=config["method_q_value_initialization"],
            config=config,
        )

    def initialize_state_values(self, config: Dict) -> Dict[State, float]:
        """Initialize the state values of the agent based on the configuration.
        This is simply a call to the `initialize_q_values` method.

        Args:
            config (Dict): the configuration of the agent

        Returns:
            Dict[State, float]: the initialized state values of the agent
        """
        return self.initialize_q_values(config=config)["whatever_state"]

    def initialize_policy_q_based(
        self,
        config: Dict,
        q_model: BaseLearner,
    ) -> PolicyQBased:
        """Initialize an ExplorativeActorQBased based on the configuration.

        Args:
            config (Dict): the configuration of the agent
            q_model (BaseLearner): the Q-values model of the agent

        Returns:
            PolicyQBased: the initialized policy of the agent
        """
        # Get the method of exploration
        assert "method_exploration" in config, (
            "The method of exploration is not specified in the configuration. Please specify it using the "
            "key 'method_exploration' in the configuration file."
        )
        self.method_exploration = config["method_exploration"]

        # Initialize the exploration method based on the method

        if self.method_exploration == "greedy":
            # Greedy exploration
            return PolicyGreedy(q_model=q_model)

        elif self.method_exploration == "eps_greedy":
            # Epsilon-greedy exploration
            assert "epsilon" in config, (
                "The epsilon value is not specified in the configuration. Please specify it using the "
                "key 'epsilon' in the configuration file."
            )
            return PolicyEpsilonGreedy(q_values=q_model, epsilon=config["epsilon"])

        elif self.method_exploration == "boltzmann":
            # Boltzmann exploration
            assert "boltzmann_temperature" in config, (
                "The temperature value is not specified in the configuration. Please specify it using the "
                "key 'temperature' in the configuration file."
            )
            return PolicyBoltzmann(
                q_values=q_model, temperature=config["boltzmann_temperature"]
            )

        elif self.method_exploration == "ucb":
            # UCB exploration
            assert "ucb_constant" in config, (
                "The UCB constant is not specified in the configuration. Please specify it using the "
                "key 'ucb_constant' in the configuration file."
            )
            return PolicyUCB(q_values=q_model, ucb_constant=config["ucb_constant"])

        else:
            raise ValueError(
                f"The method of exploration '{self.method_exploration}' is not recognized."
            )

    def get_metrics_at_transition(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool,
    ) -> Dict[str, float]:
        """Try to compute some metrics of the agent and return them as a dictionary.

        Args:
            state (State): the current state
            action (Action): the action performed
            reward (float): the reward obtained after performing the action
            next_state (State): the next state the algorithm transitions to after performing the action
            done (bool): whether the episode is over or not

        Returns:
            Dict[str, float]: the metrics of the agent
        """
        metrics = {}
        # Add Q values metrics
        if (
            try_get(self.config, "do_log_q_values", False)
            and hasattr(self, "q_values")
            and isinstance(self.q_values, dict)
        ):
            metrics.update(
                get_q_values_metrics(
                    q_values=self.q_values,
                    n_max_states_to_log=try_get(
                        self.config, "n_max_q_values_states_to_log", INF
                    ),
                )
            )
        # Add internal scheduler metrics
        if try_get(self.config, "do_log_actions_chosen", False):
            metrics.update(
                {
                    f"actions_chosen/1(A={a} in S={state})": int(a == action)
                    for a in self.q_values[state].keys()
                },
            )
        # Add internal scheduler metrics
        metrics.update(get_scheduler_metrics_of_object(obj=self))
        # Add policy's internal scheduler metrics
        if hasattr(self, "policy") and isinstance(self.policy, PolicyQBased):
            metrics.update(get_scheduler_metrics_of_object(obj=self.policy))

        return metrics
