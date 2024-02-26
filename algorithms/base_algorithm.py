# Logging
import wandb
from tensorboardX import SummaryWriter

# Config system
import hydra
from omegaconf import OmegaConf, DictConfig

# Utils
from tqdm import tqdm
import datetime
from time import time
from typing import Dict, List, Type, Any, Tuple
import cProfile

# ML libraries
import random
import numpy as np

# File specific
from abc import ABC, abstractmethod

# Project imports
from src.typing import State, Action


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
    def act(self, state: State, available_actions : List[Action], is_eval: bool = False) -> Action:
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
    ) -> None:
        """Learn from the transition `(state, action, reward, next_state)`.
        This is the method that updates the agent's knowledge.

        Args:
            state (State): the current state
            action (Action): the action performed
            reward (float): the reward obtained after performing the action
            next_state (State): the next state the algorithm transitions to after performing the action
            done (bool): whether the episode is over or not
        """
