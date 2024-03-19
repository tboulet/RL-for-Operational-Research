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
from scipy.optimize import linprog

# File specific
from abc import ABC, abstractmethod
from .base_environment import BaseOREnvironment

# Project imports
from src.typing import State, Action


class knapsack(BaseOREnvironment):

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

    def reset(self, seed=None) -> Tuple[State, dict]:
        self.weight = 0.0
        self.value = 0.0
        self.timestep = 0
        self.in_game = {i: i for i in range(self.n)}
        self.state = [0 for _ in range(self.n)]
        for i in list(self.in_game.keys()).copy():
            if self.li_objects[i]["weight_obj"] + self.weight > self.max_weight:
                self.in_game.pop(i)
        return repr(self.state), {}

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
        return repr(self.state), additional_value, False, done, {}

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
