""" Q-learning algorithm under the framework of Generalized Policy Iteration.

"""

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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn.init as init


# File specific
from abc import ABC, abstractmethod
from algorithms.base.general_policy_iterator import (
    GeneralizedPolicyIterator,
)
from src.constants import INF
from src.metrics import get_q_values_metrics, get_scheduler_metrics_of_object
from src.schedulers import get_scheduler

# Project imports
from src.typing import QValues, State, Action
from src.utils import try_get
from algorithms.base.base_algorithm import BaseRLAlgorithm

class Net(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.selu = nn.SELU()
        self.relu = nn.LeakyReLU()
        
        self.sigmo = nn.Sigmoid()

        init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x


class DeepQ_Learning(BaseRLAlgorithm):
    """Q-learning algorithm under the framework of Generalized Policy Iteration."""

    def __init__(self, config: Dict):
        BaseRLAlgorithm.__init__(self, config)
        self.epsilon = get_scheduler(self.config["epsilon"])
        self.batchsize = self.config["batch_size"]
        self.gamma = get_scheduler(self.config["gamma"])
        self.learning_rate = self.config["learning_rate"]
        self.loss = nn.HuberLoss()
        self.memory = [[{}]]
        self.tot_rewards = []
        self.size_memory = 0
        self.rewards = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        self.hidden_dim = self.config["hidden_dim"]
        self.target_time_max = self.config["target_time"]
        self.target_time = 0


    def create_net(self, input_dim: int, output_dim: int, hidden_dim :int = 128) -> nn.Module:
        self.net = Net(input_dim, output_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate, betas = (0.5,0.99))
        self.target_net = Net(input_dim, output_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())

    def act(
        self, state: State, available_actions: List[Action], is_eval: bool = False
    ) -> Action:
        if self.net is None:
            self.create_net(len(state) -1, int(state[-1].item()), hidden_dim = self.hidden_dim)
        self.net.eval()
        state = torch.tensor(state[:-1], dtype=torch.float32).unsqueeze(0)
        self.memory[-1][-1]["state"] = state
        self.memory[-1][-1]["available_actions"] = available_actions
        with torch.no_grad():
            state = state.to(self.device)
            q_values = self.net(state).squeeze().cpu().numpy()
        value = random.random()
        if value < self.epsilon.get_value():
            self.epsilon.increment_step()
            self.gamma.increment_step()
            return random.choice(available_actions)
        else:
            for i in range(len(q_values)):
                if i not in available_actions:
                    q_values[i] = -INF
            self.epsilon.increment_step()
            self.gamma.increment_step()
            return np.argmax(q_values)
    
    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool,
    ) -> Dict[str, float]:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(next_state[:-1], dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
        self.memory[-1][-1]["action"] = action
        self.memory[-1][-1]["reward"] = reward
        self.memory[-1][-1]["done"] = done
        self.memory[-1][-1]["next_state"] = next_state
        metrics = {}
        if bool(done) is False:
            self.memory[-1].append({})
        else:
            self.memory.append([{}])
        self.size_memory += 1
        self.target_time += 1
        if self.target_time > self.target_time_max:
            self.target_net.load_state_dict(self.net.state_dict())
            self.target_time = 0
        if self.size_memory >= self.batchsize:
            loss = 0
            if len(self.memory[-1]) == 1:
                self.memory.pop(len(self.memory) - 1)
            self.memory[-1].pop(len(self.memory[-1]) - 1)
            self.net.train()
            self.optimizer.zero_grad()
            for j,episode in enumerate(self.memory):
                for i, elem in enumerate(episode):
                    action, reward, done, state, available_actions = elem["action"], elem["reward"], elem["done"], elem["state"], elem["available_actions"]
                    done = bool(done)
                    q_values = self.net(state.to(self.device))
                    if bool(done) is False:
                        if (j < len(self.memory) - 1) or (i < len(episode) - 1): 
                            with torch.no_grad():
                                next_q_values = self.target_net(next_state.to(self.device))
                                next_available_actions = episode[i+1]["available_actions"]
                                for i in range(q_values.shape[1]):
                                    if i not in next_available_actions:
                                        next_q_values[0, i] = -INF
                    if bool(done) is True:
                        next_q_values = torch.zeros_like(q_values).to(self.device)   
                    target =  torch.tensor(reward).to(self.device) + torch.tensor(self.gamma.get_value()).to(self.device)* torch.max(next_q_values)
                    td_error = target - q_values[0, action]
                    loss += self.loss(q_values[:, action], target.unsqueeze(0))
            loss /= self.batchsize
            loss.backward()
            average_grad_norm = np.mean([torch.norm(param.grad).item() for param in self.net.parameters() if param.grad is not None])
            if (self.target_time == self.target_time_max -3) or (self.target_time == 0):
                print('norm',average_grad_norm, 'loss' ,loss.item())
            self.optimizer.step()
            td_error = td_error.cpu()
            target = target.cpu()
            metrics.update({"td_error": td_error.item(), "target": target.item()}) 
            self.size_memory = 0
            self.memory = [[{}]]            
        metrics.update(
            self.get_metrics_at_transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )
        )
        return metrics



