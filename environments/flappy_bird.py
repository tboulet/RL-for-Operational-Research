# Logging
import os
import sys
import wandb
from tensorboardX import SummaryWriter

# Config system
import hydra
from omegaconf import OmegaConf, DictConfig

# Utils
from tqdm import tqdm
import datetime
import time
from typing import Dict, List, Type, Any, Tuple
import cProfile

# ML libraries
import random
import numpy as np

# File specific
from abc import ABC, abstractmethod
from environments.base_environment import BaseOREnvironment
import gymnasium as gym
import text_flappy_bird_gym

# Project imports
from src.typing import State, Action


class FlappyBirdEnv(BaseOREnvironment):
    """A implementation of the Flappy Bird environment"""

    def __init__(self, config: Dict):
        """Initialize the environment with the given configuration.

        Args:
            config (Dict): the configuration of the environment
        """
        self.config = config
        self.env_string = self.config["env_string"]
        assert self.env_string in ["TextFlappyBird-v0", "TextFlappyBird-screen-v0"]
        self.env = gym.make(self.env_string, **{k : v for k, v in config.items() if k != "env_string"})

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
        state = self.env.reset()
        obs = self.observation_function(state)
        return obs, {}

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
        state, reward, done, is_trunc, info = self.env.step(action)
        obs = self.observation_function(state)
        return obs, reward, is_trunc, done, {}

    def get_available_actions(self, state : State) -> List[Action]:
        """Get the list of available actions in the current state of the environment.

        Returns:
            List[Action]: the list of available actions in the current state of the environment
        """
        return [0, 1]

    def render(self) -> None:
        """Render the environment"""
        os.system("clear")
        sys.stdout.write(self.env.render())
        time.sleep(0.2) # FPS
        
    def observation_function(self, state : State) -> State:
        if self.env_string == "TextFlappyBird-v0":
            return state[0]
        else:
            raise NotImplementedError("The observation function for the screen version of the environment is not implemented yet.")