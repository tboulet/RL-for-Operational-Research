""" SARSA algorithm under the framework of Generalized Policy Iteration.

"""

# Logging
from collections import defaultdict
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


class SARSA_Lambda(GeneralizedPolicyIterator):
    """SARSA(lambda) algorithm under the framework of Generalized Policy Iteration."""

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
        self.traces = defaultdict(lambda: defaultdict(float))
        self.lmbda = get_scheduler(config["lmbda"])
        self.do_replacing_traces = config["do_replacing_traces"]
        self.threshold_deletion_traces = config["threshold_deletion_traces"]

    def update_from_sequence_of_transitions(
        self, sequence_of_transitions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        # Hyperparameters
        gamma = self.gamma.get_value()
        # Extract the transitions
        assert len(sequence_of_transitions) == 2, "SARSA is a 2-step algorithm"
        s_t = sequence_of_transitions[0]["state"]
        a_t = sequence_of_transitions[0]["action"]
        r_t = sequence_of_transitions[0]["reward"]
        d_t = sequence_of_transitions[0]["done"]
        assert not d_t, "The sequence of transitions should not be terminal"
        s_next_t = sequence_of_transitions[1]["state"]
        a_next_t = sequence_of_transitions[1]["action"]
        d_next_t = sequence_of_transitions[1]["done"]
        # Compute TD error and update the traces
        target = r_t + gamma * self.q_model(state=s_next_t, action=a_next_t)
        self.traces[s_t][a_t] += 1
        # Update all the Q values for all the states and actions already visited, and decay the traces
        dict_averager = DictAverager()
        for s in self.traces.keys():
            a_to_remove = []
            for a in self.traces[s].keys():
                metrics_q_learner = self.q_model.learn(
                    state=s, action=a, target=target, multiplier_error=self.traces[s][a]
                )
                self.traces[s][a] *= gamma * self.lmbda.get_value()
                if self.traces[s][a] < self.threshold_deletion_traces:
                    a_to_remove.append(a)
                dict_averager.add_dict(metrics_q_learner)
                dict_averager.add("trace", self.traces[s][a])
            for a in a_to_remove:
                del self.traces[s][a]
        # In the case of replacing traces, we reset the traces to 0 if we are in a terminal state
        if self.do_replacing_traces and d_next_t:
            self.traces = defaultdict(lambda: defaultdict(float))
        # Return the metrics
        return {**dict_averager.get_dict(), "target": target}
