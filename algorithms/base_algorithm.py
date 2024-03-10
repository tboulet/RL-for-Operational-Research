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
from src.policy_q_based import PolicyEpsilonGreedy, PolicyQBased

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
        self.method_q_value_initialization = config["method_q_value_initialization"]

        # Initialize the Q-values based on the method
        if self.method_q_value_initialization == "random":
            # Random initialization : Q(s, a) ~ N(typical_return, typical_return_std)
            typical_return = try_get(config, "typical_return", 0)
            typical_return_std = try_get(config, "typical_return_std", 1)
            return defaultdict(
                lambda: defaultdict(
                    lambda: np.random.normal(typical_return, typical_return_std)
                )
            )

        elif self.method_q_value_initialization == "zero":
            # Zero initialization
            return defaultdict(lambda: defaultdict(lambda: 0.0))

        elif self.method_q_value_initialization == "optimistic":
            # Optimistic initialization
            typical_return = try_get(config, "typical_return", 0)
            typical_return_std = try_get(config, "typical_return_std", 1)
            return defaultdict(
                lambda: defaultdict(lambda: typical_return + 5 * typical_return_std)
            )

        else:
            raise ValueError(
                f"The method of Q-value initialization '{self.method_q_value_initialization}' is not recognized. Please use one of the following methods: 'random', 'zero', 'optimistic'."
            )

    def initialize_policy_q_based(
        self,
        config: Dict,
        q_values: Dict[State, Dict[Action, float]],
    ) -> PolicyQBased:
        """Initialize an ExplorativeActorQBased based on the configuration.

        Args:
            config (Dict): the configuration of the agent
            q_values (Dict[State, Dict[Action, float]]): the Q-values of the agent

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
        if self.method_exploration == "eps_greedy":
            # Epsilon-greedy exploration
            assert "epsilon" in config, (
                "The epsilon value is not specified in the configuration. Please specify it using the "
                "key 'epsilon' in the configuration file."
            )
            return PolicyEpsilonGreedy(
                q_values=q_values, epsilon=config["epsilon"]
            )

        else:
            raise ValueError(
                f"The method of exploration '{self.method_exploration}' is not recognized."
            )
