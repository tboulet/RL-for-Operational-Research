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
        self.sigmo = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.selu(self.fc1(x))
        x = self.selu(self.fc2(x))
        x = self.sigmo(self.fc3(x))
        return x


class DeepQ_Learning(BaseRLAlgorithm):
    """Q-learning algorithm under the framework of Generalized Policy Iteration."""

    def __init__(self, config: Dict):
        BaseRLAlgorithm.__init__(self, config)
        self.epsilon = get_scheduler(self.config["epsilon"])
        self.batchsize = self.config["batch_size"]
        self.gamma = get_scheduler(self.config["gamma"])
        self.learning_rate = self.config["learning_rate"]
        self.loss = nn.MSELoss()
        self.memory = [[{}]]
        self.tot_rewards = []
        self.size_memory = 0
        self.rewards = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        self.hidden_dim = self.config["hidden_dim"]


    def create_net(self, input_dim: int, output_dim: int, hidden_dim :int = 128) -> nn.Module:
        self.net = Net(input_dim, output_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
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
            # eliminer les q_values des actions qui ne sont pas dans available_actions
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
        #print(state.shape)
        next_state = torch.tensor(next_state[:-1], dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
        self.memory[-1][-1]["action"] = action
        self.memory[-1][-1]["reward"] = reward
        self.memory[-1][-1]["done"] = done
        self.memory[-1][-1]["next_state"] = next_state
        metrics = {}
        if bool(done) is False:
            self.memory[-1].append({})
            #self.rewards.append(reward)
        else:
            self.memory.append([{}])
            #self.tot_rewards.append(sum(self.rewards))
            #self.rewards = []
        self.size_memory += 1
        if self.size_memory >= self.batchsize:
            loss = 0
            self.memory[-1].pop(len(self.memory[-1]) - 1)
            for j,episode in enumerate(self.memory):
                for i, elem in enumerate(episode):
                    action, reward, done, state, available_actions = elem["action"], elem["reward"], elem["done"], elem["state"], elem["available_actions"]
                    done = bool(done)
                    #print(state.shape)
                    q_values = self.net(state.to(self.device)).cpu()
                    #print(q_values.shape)
                    if bool(done) is False:
                        with torch.no_grad():
                            next_q_values = self.net(next_state.to(self.device)).cpu()
                    if bool(done) is True:
                        next_q_values = torch.zeros_like(q_values)   
                    target = reward + self.gamma.get_value() * torch.max(next_q_values)
                    # pour toutes les actions qui ne sont pas dans available_actions, on met la target à 0
                    for i in range(len(target)):
                        if i not in available_actions:
                            target[i] = 0
                    td_error = target - q_values[0, action]
                    loss += self.loss(q_values[0, action], target)
            loss /= self.batchsize
            self.optimizer.zero_grad()
            loss.backward()
            #average_grad_norm = np.mean([torch.norm(param.grad).item() for param in self.net.parameters() if param.grad is not None])
            self.optimizer.step()
            metrics.update({"td_error": td_error.item(), "target": target.item()}) # je sais qu'il faudrait le faire après chaque calcul de loss, mais je ne sais pas comment faire
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

    def show(self):
        plt.plot(self.tot_rewards)
        plt.show()


    '''
    def update_from_sequence_of_transitions(
        self, sequence_of_transitions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        # Hyperparameters
        gamma = self.gamma.get_value()
        learning_rate = self.learning_rate.get_value()
        # Extract the transitions
        assert len(sequence_of_transitions) == 1, "Q-learning is a 1-step algorithm"
        state = sequence_of_transitions[0]["state"]
        action = sequence_of_transitions[0]["action"]
        reward = sequence_of_transitions[0]["reward"]
        done = sequence_of_transitions[0]["done"]
        next_state = sequence_of_transitions[0]["next_state"]
        # Update the Q values
        if not done and len(self.q_values[next_state]) > 0:
            target = reward + gamma * max(self.q_values[next_state].values())
        else:
            target = reward
        td_error = target - self.q_values[state][action]
        self.q_values[state][action] += learning_rate * td_error
        # Return the metrics
        return {"td_error": td_error, "target": target}
        '''
