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

# Project imports
from src.typing import State, Action


class BaseOREnvironment(gym.Env):
    """The base interface for all Operation Research environments. This class
    should be subclassed when implementing a new OR environment.
    
    It requires to implement the `reset`, `step` and `get_available_actions` methods, and suggest to implement the `render` method.
    """
    
    def __init__(self, config: Dict):
        """Initialize the environment with the given configuration.
        
        Args:
            config (Dict): the configuration of the environment
        """
        self.config = config


    @abstractmethod
    def reset(
        self,
        seed = None,
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
        
    @abstractmethod
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
        """Perform an action on the environment.

        Args:
            action (Action): the action to perform on the environment

        Returns:
            (State) : The new state of the environment
            (float) : The reward of the action
            (bool) : Whether the episode is truncated or not
            (bool) : Whether the episode is done or not
            (dict) : The info of the environment, as a dictionary
        """
        
    @abstractmethod
    def get_available_actions(self, state : State) -> List[Action]:
        """Get the list of available actions in the current state of the environment.

        Args:
            state (State): the state for which to get the list of available actions
            
        Returns:
            List[Action]: the list of available actions in the current state of the environment
        """
        
    def render(self) -> None:
        """Render the environment. This method is optional and can be implemented if needed.
        """
        pass
    
    def close(self) -> None:
        """Close the environment. This method is optional and can be implemented if needed.
        """
        pass
    
    def get_optimal_reward(self) -> float:
        """Get the optimal reward of the environment, for benchmarking purposes.
        """
        pass
    
    def get_reward_range(self) -> Tuple[float, float]:
        """Get the range of the rewards of the environment, for benchmarking purposes.
        """
        pass