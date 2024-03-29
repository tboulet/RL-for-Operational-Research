""" SARSA algorithm under the framework of Generalized Policy Iteration.

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


class SARSA(GeneralizedPolicyIterator):
    """SARSA algorithm under the framework of Generalized Policy Iteration."""

    def __init__(self, config: Dict):
        GeneralizedPolicyIterator.__init__(
            self,
            config=config,
            keys=["state", "action", "reward", "done"],
            do_terminal_learning=False,
            n_steps=2,
            do_compute_returns=False,
            do_learn_q_values=True,
            do_learn_states_values=False,
        )

    def update_from_sequence_of_transitions(
        self, sequence_of_transitions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        # Hyperparameters
        gamma = self.gamma.get_value()
        learning_rate = self.learning_rate.get_value()
        # Extract the transitions
        assert len(sequence_of_transitions) == 2, "SARSA is a 2-step algorithm"
        state = sequence_of_transitions[0]["state"]
        action = sequence_of_transitions[0]["action"]
        reward = sequence_of_transitions[0]["reward"]
        done = sequence_of_transitions[0]["done"]
        assert not done, "The sequence of transitions should not be terminal"
        next_state = sequence_of_transitions[1]["state"]
        next_action = sequence_of_transitions[1]["action"]
        # Update the Q values
        target = reward + gamma * self.q_values[next_state][next_action]
        td_error = target - self.q_values[state][action]
        self.q_values[state][action] += learning_rate * td_error
        # Return the metrics
        return {"td_error": td_error, "target": target}
