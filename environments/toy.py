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
import gym

# File specific
from abc import ABC, abstractmethod
from environments.base_environment import BaseOREnvironment

# Project imports
from src.typing import State, Action


class ToyExampleEnvironment(BaseOREnvironment):
    """A simple environment for testing purposes. The reward is 1 if a_t == s_t and 0 otherwise"""

    def __init__(self, config: Dict):
        """Initialize the environment with the given configuration.

        Args:
            config (Dict): the configuration of the environment
        """
        self.config = config
        self.n = config["n"]
        self.max_timesteps = config["max_timesteps"]
        self.timestep = None

    def reset(
        self,
        seed=None,
    ) -> Tuple[
        State,
        dict,
    ]:
        """Reset the environment to its initial state.

        Args:
            seed (int, optional): The seed to use for the random number generator if needed. Defaults to None.

        Returns:
            (State) : The initial state of the environment
            (dict) : The initial info of the environment, as a dictionary
        """
        self.state = random.choice(list(range(self.n)))
        self.timestep = 0
        return self.state, {}

    def step(
        self,
        action: Action,
    ) -> Tuple[
        State,
        float,
        bool,
        bool,
        dict,
    ]:
        """Take a step in the environment.

        Args:
            action (Action): The action to take

        Returns:
            (State) : The new state of the environment
            (float) : The reward of the action
            (bool) : Whether the episode is truncated
            (bool) : Whether the episode is done
            (dict) : The info of the environment, as a dictionary
        """
        # Check if the action is valid and the environment is in a valid state
        assert action in range(
            self.n
        ), f"Invalid action {action} for environment with {self.n} actions"
        assert self.state in range(
            self.n
        ), f"Invalid state {self.state} for environment with {self.n} states"
        assert (
            self.timestep != None
        ), "Environment was not reset, need to reset the environment"
        assert (
            self.timestep < self.max_timesteps
        ), f"Reached maximum number of timesteps {self.max_timesteps}, need to reset the environment"
        # Take a step in the environment
        reward = 1 if action == self.state else 0
        self.state = random.choice(list(range(self.n)))
        self.timestep += 1
        is_trunc = False  # 'is_trunc' will always be False for this environment
        done = (
            self.timestep >= self.max_timesteps
        )  # Environment is done once the maximum number of timesteps is reached
        # Return the transition
        return self.state, reward, is_trunc, done, {}

    def get_available_actions(self, state : State) -> List[Action]:
        """Get the list of available actions in the current state of the environment.

        Returns:
            List[Action]: the list of available actions in the current state of the environment
        """
        return list(range(self.n))

    def render(self) -> None:
        """Render the environment"""
        print(f"State: {self.state}")