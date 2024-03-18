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

    def __init__(self, config: Dict):
        # print("\n Init called\n")
        super().__init__(config)
        self.capacity = self.config["capacity"]
        self.max_size = self.config["max_size"]
        self.precision = self.config["precision"]
        self.make_optimal = self.config["optimal"]["make_optimal"]


        if not self.make_optimal:
            self.objects = np.random.uniform(0, self.max_size, self.n).round(2)
            self.n = self.config["n_items"]
        else:
            self.max_nb_objects = self.config["optimal"]["max_nb_objects"]
            self.nb_bins_optimal = self.config["optimal"]["nb_bins_optimal"]
            objects = []
            for _ in range(self.nb_bins_optimal):
                for e in self.generate_object_sizes():
                    objects.append(e) 
            self.objects = np.array(objects).round(2)
            np.random.shuffle(self.objects)
            self.n = len(self.objects)

        assert self.n > 0, "n_items must be > 0"
        assert self.capacity >= self.max_size, "capacity must be >= max_size"
        
        self.current_index = 0

        self.bins = [self.capacity]
    
    
    def reset(self, seed=None) -> Tuple[State, dict]:
        # print("\n Reset called\n")
        self.bins = [self.capacity]
        self.current_index = 0
        
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
        self.bins[action] -= self.objects[self.current_index]
        self.current_index += 1
        if self.current_index >= len(self.objects):
            done = True
        
        return self.get_format_state(), reward, truncated, done, {}

    def get_available_actions(self, state) -> List[Action]:
        # print("\n Actions called\n")
        # self.render()

        if self.current_index == self.n:
            return [None] #TODO: check this
        else:
            actions = [len(self.bins)] #+1]
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
            f"    Bins: {self.get_format_state()}\n Objects: {all_objects}\n (X=current index)"
        )  
    
    def get_optimal_reward(self) -> float:
        """Get the optimal reward of the environment, for benchmarking purposes.
        """
        return -self.nb_bins_optimal

    def get_format_state(self) -> None:
        """ This function is inplace and formats the self.bins in order to avoid 
            unprecision due to calculus which would create two different representations
            of the same state.
        """
        rounded = np.round(self.bins, self.precision)
        formatted_list = ["{:.{}f}".format(num, self.precision) for num in rounded]
        formatted_repr = "[" + ", ".join(formatted_list) + "]"
        return formatted_repr

    def generate_object_sizes(self) -> List[float]:
    
        num_objects = random.randint(1, self.max_nb_objects) 
        object_borders = np.array([random.uniform(0, self.capacity) for _ in range(num_objects)]).round(2)
        object_borders = np.insert(object_borders, 0, 0)
        object_borders = np.insert(object_borders, 0, self.capacity)
        object_borders.sort()
        object_sizes = np.diff(object_borders)

        assert np.sum(object_sizes) == self.capacity, "Sum of object sizes must be equal to capacity"

        return list(object_sizes)
    
    def round_states(self) -> None:
        self.bins = np.round(self.bins, self.precision)
        self.objects = np.round(self.objects, self.precision)
