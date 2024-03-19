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

        # Compute the L2 distances between customer sites and facility sites, as an array of shape (n_customer_sites, n_facility_sites)
        self.distances = np.linalg.norm(
            self.customer_sites[:, None, :] - self.facility_sites[None, :, :], axis=-1
        )  
    
        # Save the number of facilities to deploy
        self.n_facilities = config["n_facilities"]
        
        # Compute worst reward. Worst reward is simulated as the average cost if every facility is placed at a random location and same for customer sites, which is approximately 0.5
        self.current_cost = 0.5 * self.n_customer_sites
        self.worst_reward = -self.current_cost
                                      
        # Save the delay
        self.delay_render = config["delay_render"]

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
        self.assigned_facility_site_indexes: List[int] = []
        self.available_actions_set = set(range(self.n_facility_sites))
        self.state = repr(self.assigned_facility_site_indexes)
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
            action not in self.assigned_facility_site_indexes
        ), f"Facility site number {action} is already assigned"
        assert action in self.available_actions_set, f"Facility site number {action} is not available"
        # Assign the facility site to the facility
        self.assigned_facility_site_indexes.append(action)
        self.available_actions_set.remove(action)
        # Check if the episode is done
        done = (len(self.assigned_facility_site_indexes) == self.n_facilities)
        # Compute the reward
        new_cost = self.compute_current_cost()
        reward = self.current_cost - new_cost
        self.current_cost = new_cost
        # Compute the state
        self.state = repr(self.assigned_facility_site_indexes)
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
        return list(self.available_actions_set)

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
        if len(self.assigned_facility_site_indexes) == 0:
            facility_sites_assigned = [[], []] 
        else:
            facility_sites_assigned = list(zip(*self.facility_sites[self.assigned_facility_site_indexes]))
        plt.plot(
            facility_sites_assigned[0],
            facility_sites_assigned[1],
            "ro",
            label="Facility Sites",
        )
        
        
        # Plot
        if not self.init_render:
            plt.legend()
        plt.pause(self.delay_render)
        self.init_render = True

    def close(self) -> None:
        """Close the environment"""
        plt.close()

    def compute_current_cost(self) -> float:
        """Compute the current cost of the facility location solution

        Returns:
            float: the current_cost of the facility location solution
        """
        # Assert at least one facility is assigned
        assert len(self.assigned_facility_site_indexes) > 0, "At least one facility must be assigned when computing the current cost"
        # Get the distances matrix restricted to the assigned facility sites
        distance_matrix = self.distances[:, self.assigned_facility_site_indexes]
        # Compute the minimum distance for each customer site
        min_distances = np.min(distance_matrix, axis=1)
        # Sum the minimum distances
        return np.sum(min_distances)

    def get_optimal_reward(self) -> float:
        return
    
    def get_worst_reward(self) -> Tuple[float]:
        return