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


class Q_Learning(GeneralizedPolicyIterator):
    """Q-learning algorithm under the framework of Generalized Policy Iteration."""

    def __init__(self, config: Dict):
        GeneralizedPolicyIterator.__init__(
            self,
            config=config,
            keys=["state", "action", "reward", "next_state", "done", "prob"],
            do_terminal_learning=False,
            n_steps=1,
            do_compute_returns=False,
            do_learn_q_values=True,
            do_learn_states_values=False,
        )

    def update_from_sequence_of_transitions(
        self, sequence_of_transitions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        # Hyperparameters
        gamma = self.gamma.get_value()
        # Extract the transitions
        assert len(sequence_of_transitions) == 1, "Q-learning is a 1-step algorithm"
        s_t = sequence_of_transitions[0]["state"]
        a_t = sequence_of_transitions[0]["action"]
        r_t = sequence_of_transitions[0]["reward"]
        d_t = sequence_of_transitions[0]["done"]
        s_next_t = sequence_of_transitions[0]["next_state"]
        # Get the next Q values, depending on the type of the model
        next_q_values = self.q_model(state=s_next_t)
        next_q_values = self.get_q_values_of_state_data(q_values=next_q_values)
        # Update the Q values
        if not d_t and len(next_q_values) > 0:
            target = r_t + gamma * max(next_q_values)
        else:
            target = r_t
        metrics_q_learner = self.q_model.learn(state=s_t, action=a_t, target=target)
        # Return the metrics
        return {"target": target, **metrics_q_learner}
