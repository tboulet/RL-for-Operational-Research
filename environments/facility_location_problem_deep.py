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
from time import time, sleep
from typing import Dict, List, Literal, Type, Any, Tuple, Union
import cProfile

# ML libraries
import random
import numpy as np
import gym
from scipy.optimize import linprog

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2, InterpolationMode
import torch.nn.functional as F


# File specific
from abc import ABC, abstractmethod
from environments.base_environment import BaseOREnvironment

# Project imports
from src.typing import State, Action


class FacilityLocationProblemEnvironmentDeep(BaseOREnvironment):
    """The environment for the facility location problem.
    This environment is a version of the Facility Problem where :
    - Their is a number of known customer sites and a number of known candidate facility sites
    - Each customer site must be assigned to exactly one facility site
    - Each customer site will be assigned to the closest facility site
    - The cost is the sum of the L2 distances between each customer site and its assigned facility site
    - The goal is to minimize the cost

    Each instance of this environment represent an instance of a Facility Location Problem.
    Consequentially, during a RL training, the instance will not change, and it is only required to describe the state as the list of facility sites assigned to the facilities.

        Initialization:
        - The customer sites and facility sites are given as lists of lists of length 2, either from a list or initialized randomly if an integer is given (number of sites)
        - The number of facilities to deploy is given as an integer
        - The distance matrix between customer sites and facility sites is computed at creation of the env as an array of shape (n_customer_sites, n_facility_sites)

        State:
        - The state is represented as a list of binary values, where the i-th value is 1 if the i-th facility site is assigned to a facility, and 0 otherwise.
        - Initially, the state is only composed of 0s, as no facility site is assigned to a facility.
        - After step t, the state will contain exactly t 1s and n_facility_sites - t 0s
        - At the end of the episode, the state will contain n_facilities 1s and n_facility_sites - n_facilities 0s, as all facilities are assigned but not all facility sites are used.
        - We choose this representation as it entirely contains the information of the current state in the context of the instance of this environment object (i.e. in the instance of a Facility Location Problem).

        Actions:
        - The available actions are the facility site indexes that are not yet assigned to a facility index
        - An action 'a' corresponds to assigning the facility site number 'a' to a new facility to assign.

        Reward:
        - The reward is the loss of cost when assigning a facility site to a facility.
        - The cost is the sum of the L2 distances between each customer site and its assigned facility site.

        Termination:
        - The episode terminates when all facilities are assigned.
    """

    def __init__(self, config: Dict):
        self.config = config

        # Create the customer sites
        self.customer_sites = config["customer_sites"]
        self.customer_sites = self.get_array_of_sites(self.customer_sites)
        self.n_customers = len(self.customer_sites)
        

        # Create the facility sites
        self.facility_sites = config["facility_sites"]
        self.facility_sites = self.get_array_of_sites(self.facility_sites)
        self.n_facility_sites = len(self.facility_sites)

        # Compute the L2 distances between customer sites and facility sites, as an array of shape (n_customer_sites, n_facility_sites)
        self.distances = np.linalg.norm(
            self.customer_sites[:, None, :] - self.facility_sites[None, :, :], axis=-1
        )

        # Save the number of facilities to deploy
        self.n_facilities = config["n_facilities"]

        # Save render parameters
        self.delay_render = config["delay_render"]
        self.show_final_render = config["show_final_render"]
        self.show_lp_solution = config["show_lp_solution"]
        self.config_to_render = config["to_render"]
        
        # Save methods parameter
        self.method_reward_computation = config["method_reward_computation"]
        self.method_cost_init = config["method_cost_init"]
        
        # Compute the initial cost (dummy solution) and the worst reward
        self.initial_cost = self.compute_initial_cost(method = self.method_cost_init)
        print("Episodic initial cost computed : ", self.initial_cost)
        self.worst_reward = 0

        # Compute optimal reward
        if config["compute_lp_solution"]:
            self.optimal_reward = (
                self.compute_optimal_flp_reward() + self.initial_cost
            )
            print(f"{self.optimal_reward=}")
        else:
            self.optimal_reward = None

        # Initialize episodic variables as None
        self.are_facility_sites_assigned: List[Literal[0, 1]] = (
            None  # shape (n_facility_sites,)
        )
        self.indexes_customer_sites_to_indices_facility_sites: np.ndarray = (
            None  # shape (n_customers,)
        )
        self.lines: Dict[int, plt.Line2D] = None
        self.done = None
        self.init_render = None
        self.description = torch.cat((torch.tensor(self.customer_sites.flatten()), torch.tensor(self.facility_sites.flatten()), torch.tensor([self.n_facilities]), torch.tensor([self.n_facility_sites])))

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
        self.are_facility_sites_assigned = np.zeros(self.n_facility_sites)
        self.indexes_customer_sites_to_indices_facility_sites = None
        self.current_cost = self.initial_cost
        self.lines: List[plt.Line2D] = None
        self.init_render = False
        self.done = False
        state = torch.cat((torch.tensor(self.are_facility_sites_assigned).float(), self.description))
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
        # Check if env is initialized
        assert (
            self.are_facility_sites_assigned is not None
        ), "Environment is not initialized. Please call reset() before step()"
        # Check if the action is valid
        assert action in range(
            self.n_facility_sites
        ), f"Invalid action {action} for environment with {self.n_facility_sites} actions"
        assert (
            self.are_facility_sites_assigned[action] == 0,
        ), f"Facility site number {action} is already assigned"

        # Assign the facility site to the facility
        self.are_facility_sites_assigned[action] = 1

        # Compute the state
        state = torch.cat(( torch.tensor(self.are_facility_sites_assigned), self.description))

        # Check if the episode is done
        self.done = np.sum(self.are_facility_sites_assigned) == self.n_facilities

        # Compute the new cost according to the method_reward_computation
        if self.method_reward_computation == "step_by_step" or (
            self.method_reward_computation == "at_end" and self.done
        ):
            # Compute the new customer assignment and the new cost
            distances_with_penalty_on_non_assigned_facilities = np.copy(self.distances)
            distances_with_penalty_on_non_assigned_facilities[
                :, self.are_facility_sites_assigned == 0
            ] += np.inf
            self.indexes_customer_sites_to_indices_facility_sites = np.argmin(
                distances_with_penalty_on_non_assigned_facilities, axis=1
            )
            
            new_cost = np.sum(  # Compute the new cost, i.e. the sum of the minimum distances between each customer site and its assigned facility site
                self.distances[
                    np.arange(self.n_customers),
                    self.indexes_customer_sites_to_indices_facility_sites,
                ]
            )
        elif self.method_reward_computation == "at_end":
            new_cost = self.current_cost
        else:
            raise ValueError(
                f"Invalid method_reward_computation {self.method_reward_computation}"
            )

        # Compute the reward as the difference between the current cost and the new cost, and update the current cost
        reward = self.current_cost - new_cost
        self.current_cost = new_cost

        # Define is_trunc and info
        is_trunc = False
        info = {}

        # Return the next state, reward, is_trunc, done, and info
        return state, reward, is_trunc, self.done, info

    def get_available_actions(self, state: State) -> List[Action]:
        """Get the list of available actions (i.e. facility sites that are not yet assigned to a facility index)

        Returns:
            List[Action]: the list of available actions
        """
        return np.where(self.are_facility_sites_assigned == 0)[0]

    def render(self) -> None:
        """Render the environment"""
        
        # Print the attributes to render
        env_attributes_to_print = {key : getattr(self, key) for key, value in self.config_to_render.items() if (value and hasattr(self, key))}
        if len(env_attributes_to_print) > 0:
            print(f"Env attributes : {env_attributes_to_print}")

        # Initialize the plot if it is the first render
        if not self.init_render:
            # Create the plot
            self.fig, self.ax = plt.subplots()
            self.fig: plt.Figure
            self.ax: plt.Axes
            self.ax.set_xlabel("X-axis")
            self.ax.set_ylabel("Y-axis")
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.ax.set_title("FLP Problem : RL agent solution")
            self.ax.grid(True)
            # Plot customer sites
            customer_sites = list(zip(*self.customer_sites))
            self.ax.plot(
                customer_sites[0],
                customer_sites[1],
                "bo",
                label="Customer Sites",
                markersize=10,
            )
            self.lines: Dict[int, plt.Line2D] = {}
            # Plot facility sites
            facility_sites = list(zip(*self.facility_sites))
            self.ax.plot(
                facility_sites[0],
                facility_sites[1],
                "go",
                label="Facility Sites",
                markersize=10,
            )

        # Plot facility sites assigned
        facility_sites_assigned = self.facility_sites[
            np.where(self.are_facility_sites_assigned == 1)
        ]
        self.ax.plot(
            facility_sites_assigned[:, 0],
            facility_sites_assigned[:, 1],
            "ro",
            label="Facilities assigned",
        )

        # Plot the connections between customer sites and facility sites if the customer sites
        # Unsure that some facility sites are assigned and that customer sites are assigned to facility sites
        if (
            len(facility_sites_assigned) > 0
            and self.indexes_customer_sites_to_indices_facility_sites is not None
        ):
            assert self.lines is not None, "Lines must be initialized"
            for i in range(self.n_customers):
                if i in self.lines:
                    self.lines[i].remove()  # Remove the previous line if it exists
                (self.lines[i],) = self.ax.plot(
                    [
                        self.customer_sites[i, 0],
                        self.facility_sites[
                            self.indexes_customer_sites_to_indices_facility_sites[i], 0
                        ],
                    ],
                    [
                        self.customer_sites[i, 1],
                        self.facility_sites[
                            self.indexes_customer_sites_to_indices_facility_sites[i], 1
                        ],
                    ],
                    "k--",
                )

        # Add the legend if it is the first render and update the init_render flag
        if not self.init_render:
            self.ax.legend()
            self.init_render = True

        # Pause to see the plot and show if the episode is done
        plt.pause(self.delay_render)
        if self.done and self.show_final_render:
            sleep(5)

    def close(self) -> None:
        """Close the environment"""
        if hasattr(self, "fig"):
            plt.close(self.fig)

    # ================== Helper functions ==================

    def get_array_of_sites(self, sites: List[List[float]]) -> np.ndarray:
        """Get the array of sites from a list of lists of length 2 or an integer

        Args:
            sites (List[List[float]]): the list of sites or the number of sites, or the number of sites

        Returns:
            np.ndarray: the array of sites of shape (n_sites, 2)
        """
        if isinstance(sites, int):
            sites = np.random.rand(sites, 2)
        else:
            assert isinstance(
                sites, list
            ), "sites must be a list of lists of length 2 or an int"
            assert all(
                isinstance(site, list) and len(site) == 2 for site in sites
            ), "sites must be a list of lists of length 2 or an int"
            sites = np.array(sites)
        return sites

    def compute_initial_cost(self, method: str) -> float:
        """Compute the initial cost of a "no facility assigned" solution.
        This corresponds to the cost one would obtain with a terrible solution, and gives a baseline for the reward,
        since the reward is the loss of cost.

        Args:
            method (str): the method to compute the initial cost. Can be either "worst" or "random"

        Returns:
            float: the initial cost
        """
        if method == "max_fictive":
            # Computed as the fictive maximum cost of the solution where every customer is as far as possible from its closest facility
            # This is max_distance_in_map * n_customers
            return np.sqrt(2) * self.n_customers
        elif method == "max":
            # Computed as highest cost with one facility open.
            return np.max(np.sum(self.distances, axis=0))
        elif method == "random":
            # Computed as the cost with one random facility open.
            idx_facility_site_open = np.random.randint(self.n_facility_sites)
            return np.sum(self.distances[:, idx_facility_site_open])
        
    def compute_optimal_flp_reward(self) -> float:
        """Get the optimal reward for the Facility Location Problem, using scipy.optimize.linprog

        Returns:
            float: the optimal reward
        """
        solver = LPSolverFacilityLocationProblem(env=self)
        optimal_reward = solver.compute_lp_reward()
        if self.show_lp_solution and optimal_reward is not None:
            solver.show_lp_solution(env=self)
        return optimal_reward

    def get_optimal_reward(self) -> float:
        return self.optimal_reward

    def get_worst_reward(self) -> Tuple[float]:
        return self.worst_reward


class LPSolverFacilityLocationProblem:
    """A class for solving the Facility Location Problem using the scipy.optimize.linprog function

    After the solving, it should contains
    """

    def __init__(self, env: FacilityLocationProblemEnvironmentDeep) -> None:
        self.env = env

    def compute_lp_reward(self) -> float:
        n_customers = self.env.n_customers
        n_facility_sites = self.env.n_facility_sites
        n_facilities = self.env.n_facilities
        distances = self.env.distances
        n_variables = n_customers * n_facility_sites + n_facility_sites

        def i_j_to_xij_idx(i, j):
            return i * n_facility_sites + j

        def xij_idx_to_i_j(idx):
            return idx // n_facility_sites, idx % n_facility_sites

        def j_to_yj_idx(j):
            return n_customers * n_facility_sites + j

        def yj_idx_to_j(idx):
            return idx - n_customers * n_facility_sites

        # Define the cost vector c
        c = [None for _ in range(n_customers * n_facility_sites + n_facility_sites)]
        for i in range(n_customers):
            for j in range(n_facility_sites):
                c[i_j_to_xij_idx(i, j)] = distances[i, j]
        for j in range(n_facility_sites):
            c[j_to_yj_idx(j)] = 0.0

        # Define the constraints
        A_eq = []
        b_eq = []
        A_ub = []
        b_ub = []

        # Constraint 1 : each customer site is assigned to exactly one facility site
        for i in range(n_customers):
            a = [0.0 for _ in range(n_variables)]
            for j in range(n_facility_sites):
                a[i_j_to_xij_idx(i, j)] = 1.0
            A_eq.append(a)
            b_eq.append(1.0)
        # Constraint 2 : if a customer site is assigned to a facility site, the facility site is assigned to a facility
        for i in range(n_customers):
            for j in range(n_facility_sites):
                a = [0.0 for _ in range(n_variables)]
                a[i_j_to_xij_idx(i, j)] = 1.0
                a[j_to_yj_idx(j)] = -1.0
                A_ub.append(a)
                b_ub.append(0.0)
        # Constraint 3 : the number of active facility sites is maximum n_facilities
        a = [0.0 for _ in range(n_variables)]
        for j in range(n_facility_sites):
            a[j_to_yj_idx(j)] = 1.0
        A_ub.append(a)
        b_ub.append(n_facilities)

        # Solve the linear program
        result = linprog(
            c,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=[(0, 1) for _ in range(n_variables)],
        )
        if not result.success:
            return None

        x = result.x
        x = x[: n_customers * n_facility_sites]
        x = np.round(x).reshape(n_customers, n_facility_sites)
        self.are_facility_sites_assigned = (np.sum(x, axis=0) > 0).astype(int)
        self.indexes_customer_sites_to_indices_facility_sites = np.argmin(
            distances[:, self.are_facility_sites_assigned == 1], axis=1
        )
        return -result.fun

    def show_lp_solution(self, env: FacilityLocationProblemEnvironmentDeep) -> None:
        # Create the plot
        self.fig, self.ax = plt.subplots()
        # self.fig : plt.Figure
        # self.ax : plt.Axes
        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Y-axis")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_title("FLP Problem : LP Optimal Solution")
        self.ax.grid(True)
        # Plot customer sites
        customer_sites = list(zip(*env.customer_sites))
        self.ax.plot(
            customer_sites[0],
            customer_sites[1],
            "bo",
            label="Customer Sites",
            markersize=10,
        )
        self.lines: Dict[int, plt.Line2D] = {}
        # Plot facility sites
        facility_sites = list(zip(*env.facility_sites))
        self.ax.plot(
            facility_sites[0],
            facility_sites[1],
            "go",
            label="Facility Sites",
            markersize=10,
        )

        # Plot facility sites assigned
        facility_sites_assigned = env.facility_sites[
            np.where(self.are_facility_sites_assigned == 1)
        ]
        self.ax.plot(
            facility_sites_assigned[:, 0],
            facility_sites_assigned[:, 1],
            "ro",
            label="Facilities assigned",
        )

        # Plot the connections between customer sites and facility sites
        assert self.lines is not None, "Lines must be initialized"
        for i in range(env.n_customers):
            if i in self.lines:
                self.lines[i].remove()  # Remove the previous line if it exists
            (self.lines[i],) = self.ax.plot(
                [
                    env.customer_sites[i, 0],
                    facility_sites_assigned[
                        self.indexes_customer_sites_to_indices_facility_sites[i], 0
                    ],
                ],
                [
                    env.customer_sites[i, 1],
                    facility_sites_assigned[
                        self.indexes_customer_sites_to_indices_facility_sites[i], 1
                    ],
                ],
                "k--",
            )
        # Add the legend and show the plot
        self.ax.legend()
        self.fig.show()
