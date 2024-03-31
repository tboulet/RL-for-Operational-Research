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

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2, InterpolationMode
import torch.nn.functional as F

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


class KnapsackEnvDeep(BaseOREnvironment):

    """The environment for the Knapsack Problem
    The goal of the Knapsack Problem is to maximize the value of objects placed in a knapsack, without exceeding its maximum weight capacity.
    Each python instance of this environment represents an instance of a Knapsack Problem, so the instance caracteristics will not change during a RL training.

    Each instance of this environment represent an instance of a Facility Location Problem.
    Consequentially, during a RL training, the instance will not change, and it is only required to describe the state as the list of facility sites assigned to the facilities.

        Initialization:
        - The number of items is given as an integer, we generate a list of items with random weights and values, with uniform distribution.
        - The maximum weight of the knapsack is given as an integer. We generate a list of items with random weights and values, with uniform distribution.
        
        State:
        - The state is represented as a list of binary values, where the i-th value is 1 if the i-th item is in the knapsack, 0 otherwise.

        Actions:
        - The available actions are the item's indexes that are not yet in the knapsack and that can be added without exceeding the maximum weight of the knapsack.

        Reward:
        - The objective (undiscounted return) is to maximize the total value of the items in the knapsack.
        - Consequently, the reward is the value of the item added to the knapsack at each step.

        Termination:
        - The episode terminates when no more items can be added to the knapsack without exceeding the maximum weight.
    """
        
    def __init__(self, config: Dict):
        super().__init__(config)
        self.max_value = self.config["max_value"]
        self.max_weight = self.config["max_weight"]
        self.n = self.config["n_items"]
        self.moy_poids = self.config["moy_poids"]
        self.li_objects = [
            {
                "weight_obj": np.random.uniform(0, self.moy_poids * 2),
                "value_obj": np.random.uniform(0, self.max_value),
            }
            for _ in range(self.n)
        ]
        # Compute optimal reward
        c = [-elem["value_obj"] for elem in self.li_objects]
        A = [[elem["weight_obj"] for elem in self.li_objects]]
        b = [self.max_weight]
        res = linprog(c, A_ub=A, b_ub=b, bounds=(0, 1), integrality=np.ones(self.n))
        self.optimal_reward = -res.fun
        li_poids = [elem["weight_obj"] for elem in self.li_objects]
        li_valeurs = [elem["value_obj"] for elem in self.li_objects]
        self.description = list(li_poids + li_valeurs)  #useful for deep: type list
        self.description.append(len(li_valeurs))
        


    def reset(self, seed=None) -> Tuple[State, dict]:
        self.weight = 0.0
        self.value = 0.0
        self.timestep = 0
        self.in_game = {i: i for i in range(self.n)}
        self.state = [0 for _ in range(self.n)]
        for i in list(self.in_game.keys()).copy():
            if self.li_objects[i]["weight_obj"] + self.weight > self.max_weight:
                self.in_game.pop(i)
        return torch.cat((torch.tensor(self.state), torch.tensor(self.description))), {}

    def step(self, action: Action) -> Tuple[State, float, bool, bool, dict]:
        self.weight += self.li_objects[action]["weight_obj"]
        assert self.weight <= self.max_weight
        self.in_game.pop(action)
        additional_value = self.li_objects[action]["value_obj"]
        self.value += additional_value
        acceptable = False
        done = False
        for key in list(self.in_game.keys()).copy():
            elem = self.li_objects[key]
            if elem["weight_obj"] + self.weight <= self.max_weight:
                acceptable = True
            else:
                self.in_game.pop(key)
        if acceptable is False:
            done = True
        self.state[action] = 1
        return torch.cat((torch.tensor(self.state), torch.tensor(self.description))), additional_value, False, done, {}

    def get_available_actions(self, state) -> List[Action]:
        return list(self.in_game.keys())
        # Reprendre cette fonction pour checker les actions effectivement autorisées!!

    def render(self) -> None:
        print(
            f"Weight: {self.weight}, Value: {self.value}, State : {self.state}"
        )  # ,objets: {self.li_objects}")

    def get_optimal_reward(self) -> float:
        return self.optimal_reward

    def get_worst_reward(self) -> Tuple[float, float]:
        return 0.0
