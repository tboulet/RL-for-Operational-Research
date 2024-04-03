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
from src.metrics import DictAverager, get_q_values_metrics, get_scheduler_metrics_of_object
from src.schedulers import get_scheduler

# Project imports
from src.typing import QValues, State, Action
from src.utils import try_get
from algorithms.base.base_algorithm import BaseRLAlgorithm


class MonteCarlo(GeneralizedPolicyIterator):
    """Monte Carlo algorthm under the framework of Generalized Policy Iteration."""

    def __init__(self, config: Dict):
        GeneralizedPolicyIterator.__init__(
            self,
            config=config,
            keys=["state", "action", "reward", "done"],
            do_terminal_learning=True,
            n_steps=config["n_steps"],
            do_compute_returns=True,
            do_learn_q_values=True,
            do_learn_states_values=False,
        )

    def update_from_sequence_of_transitions(
        self, sequence_of_transitions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Update the Q values with the Monte Carlo algorithm.
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

        dict_averager = DictAverager()
        for transition in transitions_to_learn_from:
            s_t = transition["state"]
            a_t = transition["action"]
            g_t = transition["future_return"]
                        
            # Update the Q values
            metrics_q_learner = self.q_model.learn(state = s_t, action = a_t, target = g_t)
            dict_averager.add_dict(metrics_q_learner)
        
        return {**dict_averager.get_dict(), "target": g_t}