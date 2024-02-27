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


class FacilityLocationProblemEnvironment(BaseOREnvironment):
    """The environment for the facility location problem"""

    def __init__(self, config: Dict):
        self.config = config

        # Create the customer sites
        self.customer_sites = config["customer_sites"]
        if isinstance(self.customer_sites, int):
            self.customer_sites = np.random.rand(self.customer_sites, 2)
        else:
            assert isinstance(
                self.customer_sites, list
            ), "customer_sites must be a list or an int"
            assert all(
                isinstance(site, list) and len(site) == 2
                for site in self.customer_sites
            ), "customer_sites must be a list of lists of length 2"
            self.customer_sites = np.array(self.customer_sites)
        self.n_customer_sites = len(self.customer_sites)

        # Create the facility sites
        self.facility_sites = config["facility_sites"]
        if isinstance(self.facility_sites, int):
            self.facility_sites = np.random.rand(self.facility_sites, 2)
        else:
            assert isinstance(
                self.facility_sites, list
            ), "facility_sites must be a list or an int"
            assert all(
                isinstance(site, list) and len(site) == 2
                for site in self.facility_sites
            ), "facility_sites must be a list of lists of length 2"
            self.facility_sites = np.array(self.facility_sites)
        self.n_facility_sites = len(self.facility_sites)


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
        self.facility_to_assign : int  = 0
        self.facility_index_to_assigned_facility_site_index : Dict[int, int] = [None for _ in range(self.n_facility_sites)]

        state = repr(self.facility_index_to_assigned_facility_site_index)
        info = {}
        return state, info

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
        
        return self.state, reward, is_trunc, done, {}

    def get_available_actions(self, state: State) -> List[Action]:
        """Get the list of available actions (i.e. facility sites that are not yet assigned to a facility index)

        Returns:
            List[Action]: the list of available actions
        """
        return [i for i in range(self.n_facility_sites) if self.facility_index_to_assigned_facility_site_index[i] is None]

    def render(self) -> None:
        """Render the environment"""
        print(f"State: {self.state}")
