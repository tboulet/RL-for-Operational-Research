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
from copy import deepcopy

# ML libraries
import random
import numpy as np
import gym
from scipy.optimize import linprog

# File specific
from abc import ABC, abstractmethod
from .base_environment import BaseOREnvironment

# Project imports
from src.typing import State, Action


class BinPacking(BaseOREnvironment):
    """The environment for the bin packing problem
    This environment is a version of the bin packing problem where :
    - Each bins as a maximum capacity of config["capacity"]
    - Each bins is filled with objects of random sizes, with a maximum number of objects of config["max_nb_objects"]
    - The reward is the number of bins used to store all objects. It is negative, as the goal is to minimize the number of bins used.
    - The state is represented as a list of the remaining capacity of each bin.
    - Due to computatational errors from Python, the state is rounded to a precision of config["precision"].

    Each instance of this environment represents an instance of a bin packing problem.
    Consequentially, during a RL training, the instance will not change, and so will the objects to place.

        Initialization:
        - The capacity of the bins is given as an integer, we make a list of bins with this capacity
        - The list of objects to place is generated at the beginning of the episode, with random sizes and a maximum number of objects.
        - Thus, all objects have a size between 0 and the capacity of the bins.
        - The list of objects is shuffled at the beginning of the episode in order to prevent the agent from learning a specific order of objects to place..

        State:
        - The state is represented as a list of the remaining capacity of each bin. The last cell of the list as full capacity, for the action "create a new bin".
        - All floats are rounded to a precision of config["precision"].
        - At the end of the episode, the state will have a length of the number of bins used.
        - Its maximum length is the number of objects to place, the minimum length is nb_bins_optimal.

        Actions:
        - At each step i, the agent can choose to assign the object in the i-th position to a bin, or create a new bin.
        - Deciding to put hte object in a bin is based in the remaining capacity of the bin and the size of the object.

        Reward:
        - The reward is the number of bins used to store all objects. It is negative, as the goal is to minimize the number of bins used.
        - At every step, the agent receives a reward of -1 if it puts a objets to a new bin, otherwise it receives the reward 0.

        Termination:
        - The episode terminates at step n, when all objects have been placed in a bin.
    """

    def __init__(self, config: Dict):

        super().__init__(config)
        self.capacity = self.config["capacity"]
        self.precision = self.config["precision"]
        self.max_nb_objects = self.config["max_nb_objects"]
        self.nb_bins_optimal = self.config["nb_bins_optimal"]

        # self.max_size = self.config["max_size"] # deprecated right now
        # self.make_optimal = self.config["make_optimal"] # deprecated right now

        objects = []
        for _ in range(self.nb_bins_optimal):
            for e in self.generate_object_sizes():
                objects.append(e)
        self.objects = np.array(objects).round(2)
        np.random.shuffle(self.objects)
        self.n = len(self.objects)

        # save self.bins which is a list 
        assert ( np.isclose(np.sum(self.objects), self.capacity * self.nb_bins_optimal, rtol=1e-5)
        ), "Sum of object sizes must be equal to capacity * nb_bins_optimal"

        assert self.n > 0, "n_items must be > 0"
        # assert self.capacity >= self.max_size, "capacity must be >= max_size"

    def reset(self, seed=None) -> Tuple[State, dict]:
        # print("\n Reset called\n")
        self.bins = []
        self.current_index = 0
        self.reward_render = 0 #Used for rendering only

        return repr(self.bins), {}

    def step(self, action: Action) -> Tuple[State, float, bool, bool, dict]:
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
        # print("\nStep called\n")
        # self.render()
        done, truncated = False, False
        reward = 0.0

        if action == len(self.bins):
            self.bins.append(self.capacity)
            reward = -1.0
            self.reward_render += reward
        self.bins[action] -= self.objects[self.current_index]
        self.current_index += 1
        if self.current_index >= len(self.objects):
            done = True
        self.bins = list(np.sort(self.bins)[::-1])
        return self.get_format_state(), reward, truncated, done, {}

    def get_available_actions(self, state) -> List[Action]:
        # print("\n Actions called\n")
        # self.render()

        if self.current_index == self.n:
            return [None]  # TODO: check this
        else:
            actions = [len(self.bins)]  # +1]
            # size_next_object = self.objects[self.current_index + 1]
            size_next_object = self.objects[self.current_index]
            for i in range(len(self.bins)):
                if size_next_object <= self.bins[i]:
                    actions.append(i)
            # print(f"\nActions:{actions} / {size_next_object} ")
            return actions

    def render(self) -> None:
        all_objects = list(deepcopy(self.objects))
        all_objects.insert(self.current_index, "X")
        print(
            f"    Bins: {self.get_format_state()}\n Objects: {all_objects}\n (X=current index)\
                \n possible actions: {self.get_available_actions(self.get_format_state())}, current reward {self.reward_render} \n"
        )

    def get_optimal_reward(self) -> float:
        """Get the optimal reward of the environment, for benchmarking purposes."""
        return -self.nb_bins_optimal

    def get_worst_reward(self) -> float:
        """Get the worst reward of the environment, for benchmarking purposes."""
        return -2 * self.nb_bins_optimal

    def get_format_state(self) -> None:
        """This function is inplace and formats the self.bins in order to avoid
        unprecision due to calculus which would create two different representations
        of the same state.
        """
        rounded = np.round(self.bins, self.precision)
        formatted_list = ["{:.{}f}".format(num, self.precision) for num in rounded]
        formatted_repr = "[" + ", ".join(formatted_list) + "]"
        return formatted_repr

    def generate_object_sizes(self) -> List[float]:

        num_objects = random.randint(1, self.max_nb_objects)
        object_borders = np.array(
            [random.uniform(0, self.capacity) for _ in range(num_objects - 1)]
        ).round(2)
        object_borders = np.insert(object_borders, 0, 0)
        object_borders = np.insert(object_borders, 0, self.capacity)
        object_borders = np.unique(object_borders)
        object_borders.sort()
        object_sizes = np.diff(object_borders)
        object_sizes[-1] = self.capacity - sum(object_sizes[:-1])

        assert (
            np.isclose(np.sum(object_sizes), self.capacity, rtol=1e-5),
        ), "Sum of object sizes must be equal to capacity"
        assert 0 not in object_sizes, "No object size can be equal to 0"

        return np.around(object_sizes, decimals=2).tolist()
    
    def round_states(self) -> None:
        self.bins = np.round(self.bins, self.precision)
        self.objects = np.round(self.objects, self.precision)
