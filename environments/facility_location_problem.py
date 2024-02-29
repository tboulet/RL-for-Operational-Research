# Logging
from matplotlib import pyplot as plt
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

        # Save the number of facilities to deploy
        self.n_facilities = config["n_facilities"]

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
        self.facility_to_assign: int = 0
        self.facility_index_to_assigned_facility_site_index: List[int] = [
            None for _ in range(self.n_facilities)
        ]

        self.state = repr(self.facility_index_to_assigned_facility_site_index)
        info = {}
        self.init_render = False
        return self.state, info

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
        # Check if the action is valid
        assert action in range(
            self.n_facility_sites
        ), f"Invalid action {action} for environment with {self.n_facility_sites} actions"
        assert (
            self.facility_index_to_assigned_facility_site_index[self.facility_to_assign]
            is None
        ), f"Facility number {self.facility_to_assign} is already assigned"
        assert (
            action not in self.facility_index_to_assigned_facility_site_index
        ), f"Facility site number {action} is already assigned"
        # Assign the facility site to the facility
        self.facility_index_to_assigned_facility_site_index[self.facility_to_assign] = (
            action
        )
        self.facility_to_assign += 1
        # Check if the episode is done
        done = self.facility_to_assign == self.n_facilities
        # Compute the reward
        reward = 0
        if done:
            # Compute the reward
            reward = self.compute_reward()
        # Compute the state
        self.state = repr(self.facility_index_to_assigned_facility_site_index)
        # Define is_trunc and info
        is_trunc = False
        info = {}
        # Return the next state, reward, is_trunc, done, and info
        return self.state, reward, is_trunc, done, info

    def get_available_actions(self, state: State) -> List[Action]:
        """Get the list of available actions (i.e. facility sites that are not yet assigned to a facility index)

        Returns:
            List[Action]: the list of available actions
        """
        return [
            i
            for i in range(self.n_facility_sites)
            if i not in self.facility_index_to_assigned_facility_site_index
        ]

    def render(self) -> None:
        """Render the environment"""
        if not self.init_render:
            plt.figure(figsize=(8, 6))
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.title("Facility and Customer Sites")
            plt.grid(True)
            

        # Plot customer sites
        customer_sites = list(zip(*self.customer_sites))
        plt.plot(customer_sites[0], customer_sites[1], 'bo', label='Customer Sites')

        # Plot facility sites
        facility_sites_assigned_indexes = [_ for _ in self.facility_index_to_assigned_facility_site_index if _ is not None]
        facility_sites_assigned = self.facility_sites[facility_sites_assigned_indexes]
        facility_sites_assigned = list(zip(*facility_sites_assigned))  
        if len(facility_sites_assigned) == 0:
            facility_sites_assigned = [[], []]      
        plt.plot(
            facility_sites_assigned[0],
            facility_sites_assigned[1],
            "ro",
            label="Facility Sites",
        )
        
        
        # Plot
        if not self.init_render:
            plt.legend()
        plt.pause(0.1)
        self.init_render = True

    def close(self) -> None:
        """Close the environment"""
        plt.close()

    def compute_reward(self) -> float:
        """Compute the reward of the environment

        Returns:
            float: the reward of the environment
        """
        # Compute the distance between the facility site and the customer sites
        facility_sites_assigned = self.facility_sites[
            self.facility_index_to_assigned_facility_site_index
        ]
        distance_matrix = np.linalg.norm(
            facility_sites_assigned[:, None, :] - self.customer_sites[None, :, :],
            axis=-1,
        )
        # Compute the reward
        reward = -np.sum(np.min(distance_matrix, axis=0))
        return reward
