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
from src.constants import INF
from src.metrics import get_q_values_metrics, get_scheduler_metrics_of_object
from src.schedulers import get_scheduler

# Project imports
from src.typing import QValues, State, Action
from src.utils import try_get
from .base_algorithm import BaseRLAlgorithm


class SARSA(BaseRLAlgorithm):
    """An implementation of the SARSA algorithm for reinforcement learning."""

    def __init__(self, config: Dict):
        super().__init__(config)

        # Hyperparameters
        self.gamma = config["gamma"]
        self.learning_rate = get_scheduler(config["learning_rate"])

        # Initialize the Q values
        self.q_values: QValues = self.initialize_q_values(config=config)

        # Initialize the exploration method
        self.policy = self.initialize_policy_q_based(
            config=config, q_values=self.q_values
        )

        # Initialize the "memory" of the agent, here only the last transitions
        self.has_done_at_least_one_episode_step = False
        self.last_transition: Tuple[State, Action, float, State, bool] = None

    def act(
        self, state: State, available_actions: List[Action], is_eval: bool = False
    ) -> Action:
        # If the agent is not evaluating, we update the scheduler(s)
        if not is_eval:
            self.learning_rate.increment_step()
        # Choose the action
        action, prob = self.policy.get_action_and_prob(
            state=state,
            available_actions=available_actions,
            is_eval=is_eval,
        )
        # # Eventually save the probability of the action  # Example for algorithms that use probabilities
        # self.last_prob = prob
        return action

    def update(
        self, state: State, action: Action, reward: float, next_state: State, done: bool
    ) -> Dict[str, float]:

        # We get the learning rate
        alpha = self.learning_rate.get_value()

        # If we have done at least one step in the episode, we can do the update of the previous transition
        if self.has_done_at_least_one_episode_step:
            # Unpack the last transition
            assert (
                self.last_transition is not None
            ), "The last transition should not be None"
            last_state, last_action, last_reward, last_next_state, last_done = (
                self.last_transition
            )
            assert (
                state == last_next_state
            ), "The current state should be the next state of the last transition"
            assert (
                not last_done
            ), "The last transition should not be done, otherwise the episode is done and we wouldn't be here"

            # Get the target
            if not last_done:
                target = last_reward + self.gamma * self.q_values[state][action]
            else:
                target = last_reward

            # Compute SARSA's Temporal Difference error
            td_error = target - self.q_values[last_state][last_action]

            # Update the Q value
            self.q_values[last_state][last_action] += alpha * td_error

        # If done, we do the update on the current transition now.
        # And we reset the memory as this is a new episode.
        if done:
            td_error = reward - self.q_values[state][action]
            self.q_values[state][action] += alpha * td_error
            self.has_done_at_least_one_episode_step = False
            self.last_transition = None

        # Else, we save the transition for the next update.
        else:
            self.has_done_at_least_one_episode_step = True
            self.last_transition = (state, action, reward, next_state, done)

        # Log the metrics
        metrics = {}
        if try_get(self.config, "do_log_q_values", False):
            metrics.update(
                get_q_values_metrics(
                    q_values=self.q_values,
                    n_max_states_to_log=try_get(
                        self.config, "n_max_q_values_states_to_log", INF
                    ),
                )
            )
        if try_get(self.config, "do_log_actions_chosen", False):
            metrics.update(
                {
                    f"actions_chosen/1(A={a} in S={state})": int(a == action)
                    for a in self.q_values[state].keys()
                },
            )
        metrics.update(get_scheduler_metrics_of_object(obj=self))
        try:
            metrics["TD error"] = td_error
        except:
            pass
        return metrics
