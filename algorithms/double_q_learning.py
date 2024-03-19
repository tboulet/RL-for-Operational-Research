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


class DoubleQ_Learning(GeneralizedPolicyIterator):
    """Double Q-learning algorithm under the framework of Generalized Policy Iteration."""

    def __init__(self, config: Dict):
        GeneralizedPolicyIterator.__init__(
            self,
            config=config,
            keys=["state", "action", "reward", "next_state", "done"],
            do_terminal_learning=False,
            n_steps=1,
            do_compute_returns=False,
            do_learn_q_values=True,
            do_learn_states_values=False,
        )

        # Define Q1 and Q2 so that Q1 + Q2 = Q
        self.q_values_1 = self.initialize_q_values(config=config)
        self.q_values_2 = self.initialize_q_values(config=config)
        for s in self.q_values:
            for a in self.q_values[s]:
                self.q_values_2[s][a] = (
                    self.q_values[s][a] - self.q_values_1[s][a]
                )  # this initialize q1[s][a] and q2[s][a] such that q1[s][a] + q2[s][a] = q[s][a]

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
        if (
            not done
            and len(self.q_values_1[next_state]) > 0
            and len(self.q_values_2[next_state]) > 0
        ):
            best_q2_action = max(
                self.q_values_2[next_state], key=self.q_values[next_state].get
            )
            target_2 = reward + gamma * self.q_values_1[next_state][best_q2_action]
            best_q1_action = max(
                self.q_values_1[next_state], key=self.q_values_1[next_state].get
            )
            target_1 = reward + gamma * self.q_values_2[next_state][best_q1_action]
        else:
            target_1 = reward
            target_2 = reward

        td_error_1 = target_1 - self.q_values_1[state][action]
        td_error_2 = target_2 - self.q_values_2[state][action]
        self.q_values_1[state][action] += learning_rate * td_error_1
        self.q_values_2[state][action] += learning_rate * td_error_2
        self.q_values[state][action] = 0.5 * (
            self.q_values_1[state][action] + self.q_values_2[state][action]
        )

        # Return the metrics
        return {"td_error": td_error_1, "target": target_1}
