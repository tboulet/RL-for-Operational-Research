""" n-step SARSA algorithm under the framework of Generalized Policy Iteration.

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


class n_step_SARSA(GeneralizedPolicyIterator):
    """n-step SARSA algorithm under the framework of Generalized Policy Iteration."""

    def __init__(self, config: Dict):
        self.n_steps_for_sarsa = config["n_steps"]
        GeneralizedPolicyIterator.__init__(
            self,
            config=config,
            keys=["state", "action", "reward", "done"],
            do_terminal_learning=False,
            n_steps=self.n_steps_for_sarsa + 1,
            do_compute_returns=False,
            do_learn_q_values=True,
            do_learn_states_values=False,
        )

    def update_from_sequence_of_transitions(
        self, sequence_of_transitions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        # Extract the transitions
        assert (
            len(sequence_of_transitions) <= self.n_steps_for_sarsa + 1
        ), f"{self.n_steps_for_sarsa}-step SARSA is a {self.n_steps_for_sarsa+1}-step algorithm, but we received a sequence of transitions of length {len(sequence_of_transitions)} (too long). Sequence : {sequence_of_transitions}"
        assert (
            len(sequence_of_transitions) == self.n_steps_for_sarsa + 1
            or sequence_of_transitions[-1]["done"]
        ), f"n-step SARSA is a {self.n_steps_for_sarsa+1}-step algorithm, but we received a non terminal sequence of transitions of length {len(sequence_of_transitions)} (too short). Sequence : {sequence_of_transitions}"

        s_t = sequence_of_transitions[0]["state"]
        a_t = sequence_of_transitions[0]["action"]
        
        # If not done, we do the n-step SARSA update : X_t = R_t + g * R_{t+1} + ... + g^{n-1} * R_{t+n-1} + g^n * Q(S_{t+n}, A_{t+n})
        if "previous_target" not in sequence_of_transitions[0]:
            target = self.compute_n_step_sarsa_target(
                sequence_of_transitions=sequence_of_transitions,
                n_steps_for_sarsa=self.n_steps_for_sarsa,
            )
        else:
            raise NotImplementedError(
                "The previous_target should not be in the sequence_of_transitions"
            )  # TODO : implement the case where the previous_target is in the sequence_of_transitions

        # Update the Q values
        metrics_q_learner = self.q_model.learn(state=s_t, action=a_t, target=target)

        # Return the metrics
        return {"target": target, **metrics_q_learner}
