""" Monte Carlo algorithm under the framework of Generalized Policy Iteration.

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
from src.policies.policy_differentiable import PolicyTabular
from src.schedulers import get_scheduler

# Project imports
from src.typing import QValues, State, Action
from src.utils import try_get
from algorithms.base.base_algorithm import BaseRLAlgorithm


class REINFORCE(GeneralizedPolicyIterator):
    """REINFORCE algorthm under the framework of Generalized Policy Iteration."""

    def __init__(self, config: Dict):

        self.policy = PolicyTabular()

        GeneralizedPolicyIterator.__init__(
            self,
            config=config,
            keys=["state", "action", "reward", "done"],
            do_terminal_learning=True,
            n_steps=config["n_steps"],
            do_compute_returns=True,
            do_learn_q_values=False,
            do_learn_states_values=False,
            is_policy_q_based=False,
        )

    def update_from_sequence_of_transitions(
        self, sequence_of_transitions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Learn from a sequence of transitions with the REINFORCE algorithm.
        We receive a sequence of transitions of len n_steps (or less if the sequence is terminal), consequentially we learn from :
        - the first transition
        - if the sequence is terminal, the other transitions

        Args:
            sequence_of_transitions (List[Dict[str, Any]]): the sequence of transitions

        Returns:
            Dict[str, float]: the metrics of the update
        """
        sequence_is_terminal = sequence_of_transitions[-1]["done"]
        if not sequence_is_terminal:
            transitions_to_learn_from = sequence_of_transitions[0:1]
        else:
            transitions_to_learn_from = sequence_of_transitions

        alpha = self.learning_rate.get_value()
        sum_advantage_value = 0
        for transition in transitions_to_learn_from:
            s_t = transition["state"]
            a_t = transition["action"]
            g_t = transition["future_return"]
            self.policy.gradient_step(
                state=s_t,
                action=a_t,
                learning_rate=alpha,
                advantage=g_t,
            )
            sum_advantage_value += g_t
        return {"Advantage function": sum_advantage_value / len(transitions_to_learn_from)}
