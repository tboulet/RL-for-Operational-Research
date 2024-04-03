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
from src.learners.base_learner import BaseQValuesLearner
from src.metrics import get_q_values_metrics, get_scheduler_metrics_of_object
from src.schedulers import get_scheduler

# Project imports
from src.typing import QValues, State, Action
from src.utils import instantiate_class, try_get
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

        # We define here Q1, and we will define Q2 as Q - Q1
        self.q_model_1 : BaseQValuesLearner = instantiate_class(
                **config["q_model"],
                learning_rate=self.learning_rate,
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
        if isinstance(next_q_values, dict):
            next_q_values_1 = self.q_model_1(state=s_next_t)
            next_q_values_2 = {a : next_q_values[a] - next_q_values_1[a] for a in next_q_values}
        elif isinstance(next_q_values, np.ndarray):
            next_q_values_1 = self.q_model_1(state=s_next_t)
            next_q_values_2 = next_q_values - next_q_values_1
        else:
            raise ValueError("The type of next_q_values is not recognized")
        
        # Update the Q values
        if (
            not d_t
            and len(next_q_values_1) > 0
            and len(next_q_values_2) > 0
        ):
            # Get the best action from Q1, depending on the type of the model
            if isinstance(next_q_values_1, dict):
                best_q1_action = max(
                    next_q_values_1, key=next_q_values_1.get
                )
                target_2 = r_t + gamma * next_q_values_2[best_q1_action]
                
                best_q2_action = max(
                    next_q_values_2, key=next_q_values_2.get
                )
                target_1 = r_t + gamma * next_q_values_1[best_q2_action]
                
            elif isinstance(next_q_values_1, np.ndarray):
                best_q1_action = np.argmax(next_q_values_1)
                target_2 = r_t + gamma * next_q_values_2[best_q1_action]
                
                best_q2_action = np.argmax(next_q_values_2)
                target_1 = r_t + gamma * next_q_values_1[best_q2_action]
            
            else:
                raise ValueError("The type of next_q_values_1 is not recognized")
                            
        else:
            target_1 = r_t
            target_2 = r_t

        target = target_1 + target_2
        metrics_q_learner_1 = self.q_model_1.learn(state=s_t, action=a_t, target=target_1)
        metrics_q_learner = self.q_model.learn(state=s_t, action=a_t, target=target)
        # Return the metrics
        return {"target": target, **metrics_q_learner}