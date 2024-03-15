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

# File specific
from abc import ABC, abstractmethod

# Project imports
from src.typing import State, Action
from .base_algorithm import BaseRLAlgorithm
from itertools import product

class Q_learning(BaseRLAlgorithm):

    def __init__(self, config: Dict):
        super().__init__(config)
        self.value = None
        self.Q = {}
        self.eps = self.config["eps"]
        self.alpha = self.config["alpha"]
        self.min_Q = self.config["min_Q_init"]
        self.max_Q = self.config["max_Q_init"]




    def act(self, state: State, available_actions: List[Action], is_eval: bool = False) -> Action:
        if str(state) not in self.Q:
            if self.value is None:
                self.Q[str(state)] = {action: np.random.uniform(self.min_Q,self.max_Q) \
                                for action in available_actions}
            else:
                self.Q[str(state)] = {action: self.value for action in available_actions}
        if np.random.uniform() < self.eps:
            return random.choice(available_actions)
        Q_state = self.Q[str(state)]
        best_val = -np.inf 
        best_act = []
        for action in available_actions:
            if Q_state[action] > best_val:
                best_act = [action]
                best_val = Q_state[action] 
            if Q_state[action] == best_val:
                best_act.append(action)
        return random.choice(best_act)
        
    def update(self, state: State, action: Action, reward: float, next_state: State, done: bool) -> Dict[str, float]:
        if done:
            self.Q[str(state)][action] = (1-self.alpha)*self.Q[str(state)][action] + self.alpha*reward
            self.value = None
        else:
            if str(next_state) not in self.Q:
                self.value = np.random.uniform(self.min_Q,self.max_Q)
            else:
                self.value = max(self.Q[str(next_state)].values())
            self.Q[str(state)][action] = (1-self.alpha)*self.Q[str(state)][action] + self.alpha*(reward + self.value)
        return {}
        return {
            **{f"Q(s={s}, a={a})": self.Q[s][a] for s in self.Q for a in self.Q[s]},
            **{f"1(A={a} in S={state})": int(a == action) for a in self.Q[str(state)].keys()},
        }
        
            




    