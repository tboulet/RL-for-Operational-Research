""" Base file for the learners object.
Learners are python objects that learn a target function from examples iteratively.

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
from typing import Dict, List, Optional, Type, Any, Tuple, Union
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
from src.initialization import initialize_tabular_q_values
from src.learners.base_learner import BaseLearner, BaseQValuesLearner
from src.metrics import get_q_values_metrics, get_scheduler_metrics_of_object
from src.schedulers import Scheduler, get_scheduler

# Project imports
from src.typing import QValues, State, Action
from src.utils import try_get
from algorithms.base.base_algorithm import BaseRLAlgorithm


class QValuesTabularLearner(BaseQValuesLearner):
    """A tabular learner for the Q values."""

    def __init__(
        self,
        learning_rate: Scheduler,
        method_q_value_initialization: str,
        typical_return: Optional[float] = 0,
        typical_return_std: Optional[float] = 1,
    ) -> None:
        super().__init__(learning_rate)
        self.q_table = initialize_tabular_q_values(
            method_q_value_initialization=method_q_value_initialization,
            typical_return=typical_return,
            typical_return_std=typical_return_std,
        )

    def learn(
        self,
        state: State,
        action: Action,
        target: Union[float, int, np.ndarray],
        multiplier_error: float = 1,
    ) -> Dict[str, float]:
        alpha = self.learning_rate.get_value()
        error = target - self.q_table[state][action]
        self.q_table[state][action] += alpha * error * multiplier_error
        return {
            "TD error": error,
            "target": target,
            "q_value": self.q_table[state][action],
        }

    def __call__(
        self, **conditional_variables: List[Any]
    ) -> Union[float, int, np.ndarray]:
        assert (
            len(conditional_variables) <= 2
        ), "The QValuesTabularLearner takes at most 2 conditional variables : state and action."
        if len(conditional_variables) == 1:
            assert (
                "state" in conditional_variables
            ), "If there is only one conditional variable, it should be the state."
            return self.q_table[conditional_variables["state"]]
        elif len(conditional_variables) == 2:
            assert (
                "state" in conditional_variables and "action" in conditional_variables
            ), "If there are two conditional variables, they should be the state and the action."
            state = conditional_variables["state"]
            action = conditional_variables["action"]
            return self.q_table[state][action]
        else:
            raise ValueError(
                "The QValuesTabularLearner takes at most 2 conditional variables : state and action."
            )

    def get_available_actions(self, state: State) -> List[Action]:
        return list(self.q_table[state].keys())


class StateValuesTabularLearner:
    pass
