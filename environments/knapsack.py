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
from .base_environment import BaseOREnvironment

# Project imports
from src.typing import State, Action

class knapsack(BaseOREnvironment):

    def __init__(self,config: Dict):
        super().__init__(config)
        self.max_value = self.config["max_value"]
        self.max_weight = self.config["max_weight"]
        self.n = self.config["n_items"]
        self.rerandom = self.config["rerandom"]   #définit si on garde toujours le même sac
        self.exists = False #définit si on a déjà créé un sac
        self.moy_poids = self.config['moy_poids']


    def reset(self, seed=None) -> Tuple[State, dict]:
        self.new_state = None
        self.weight = 0.0
        self.value = 0.0
        self.timestep = 0
        if (self.rerandom is True) or (self.exists is False):
            self.exists = True
            self.li_objects = [{'weight_obj':np.random.uniform(0,self.moy_poids*2),\
                                'value_obj': np.random.uniform(0,self.max_value) }   for _ in range(self.n)]
        self.in_game = {i:i for i in range(self.n)} 
        self.state =  [0 for _ in range(self.n)]
        for i in list(self.in_game.keys()).copy():
            if self.li_objects[i]['weight_obj'] + self.weight > self.max_weight:
                self.in_game.pop(i)
        return self.state, {}
    
    def step(self, action: Action) -> Tuple[State, float, bool,bool, dict]:
        if self.new_state is not None:
            self.state = self.new_state.copy()
        self.weight += self.li_objects[action]['weight_obj']
        assert self.weight <= self.max_weight
        self.in_game.pop(action)
        self.value += self.li_objects[action]['value_obj']
        acceptable = False
        done = False
        for key in list(self.in_game.keys()).copy():
            elem = self.li_objects[key]
            if elem['weight_obj'] + self.weight <= self.max_weight:
                acceptable = True
            else:
                self.in_game.pop(key)
        if acceptable is False:
            done = True
        self.new_state = self.state.copy()
        self.new_state[action] = 1
        return self.new_state, self.value, False, done, {}
    
    def get_available_actions(self,state) -> List[Action]:
        return list(self.in_game.keys())
        #Reprendre cette fonction pour checker les actions effectivement autorisées!!
    
    def render(self) -> None:
        print(f"Weight: {self.weight}, Value: {self.value}, State : {self.state},objets: {self.li_objects}")

            

            

            

