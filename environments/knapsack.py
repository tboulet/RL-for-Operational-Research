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

class knapsack(BaseOREnvironment):

    def __init__(self,config: Dict):
        super().__init__(config)
        self.max_value = self.config["max_value"]
        self.max_weight = self.config["max_weight"]
        self.n = self.config["n_items"]

    def reset(self, seed=None) -> Tuple[State, dict]:
        self.weight = 0.0
        self.value = 0.0
        self.state = ([])
        self.timestep = 0
        self.li_objects = [{'weight_obj':np.random.uniform(0,self.max_weight/2),\
                            'value_obj': np.random.uniform(0,self.max_value) }   for _ in range(self.n)]
        return self.state, {}
    
    def step(self, action: Action) -> Tuple[State, float, bool,bool, dict]:
        acceptable = False
        n_ite = 0
        while (acceptable is False) and n_ite <= self.n -1:
            elem = self.li_objects[n_ite]
            if elem['weight_obj'] + self.weight <= self.max_weight:
                acceptable = True
        if acceptable is False:
            done = True
            return self.state, self.value, False, done, {}
        else:
            pass

            

