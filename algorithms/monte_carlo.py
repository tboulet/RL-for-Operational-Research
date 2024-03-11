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
from algorithms.algorithms_n_steps import AlgorithmNSteps
from src.constants import INF
from src.metrics import get_q_values_metrics, get_scheduler_metrics_of_object
from src.schedulers import get_scheduler

# Project imports
from src.typing import QValues, State, Action
from src.utils import try_get
from .base_algorithm import BaseRLAlgorithm


class MonteCarlo(AlgorithmNSteps):
    """An implementation of the Monte Carlo algorithm for reinforcement learning.

    It does the following :
    - Play a full episode
    - Then, update the Q values at the end of the episode
    - For this, for each state q-state (s,a) seen in the episode, we update the Q value of (s,a) towards the discounted return
    """

    def __init__(self, config: Dict):

        # Hyperparameters
        self.gamma = get_scheduler(config["gamma"])
        self.learning_rate = get_scheduler(config["learning_rate"])
        self.n_steps = config["n_steps"]

        # Initalize the AlgorithmNSteps class, which provides this object with a memory object
        AlgorithmNSteps.__init__(
            self,
            config=config,
            keys=["state", "action", "reward", "next_state", "done"],
            n_steps=self.n_steps,
            do_compute_returns=True,
            gamma=self.gamma,
        )

        # Initialize the Q values
        self.q_values: QValues = self.initialize_q_values(config=config)

        # Initialize the exploration method
        self.policy = self.initialize_policy_q_based(
            config=config, q_values=self.q_values
        )

    def act(
        self, state: State, available_actions: List[Action], is_eval: bool = False
    ) -> Action:
        # If the agent is not evaluating, we update the scheduler(s)
        if not is_eval:
            self.learning_rate.increment_step()
            self.gamma.increment_step()
        # Choose the action
        action, prob = self.policy.act(
            state=state,
            available_actions=available_actions,
            is_eval=is_eval,
        )
        return action

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool,
    ) -> Dict[str, float]:

        # We store the transition in the memory
        self.handle_transition_for_memory(
            transition={
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
            }
        )

        # We compute here the indexes from which we will update the Q values. We will update :
        # - if the memory is full, we will update from the first transition (because it is gonna be removed from the memory after the update)
        # - if we have reached a terminal state, we will update from all transitions and we will reset the memory
        first_step_of_memory = self.memory.steps_in_memory[0]
        last_step_of_memory = self.memory.steps_in_memory[-1]
        if not done:
            if self.memory.is_full():
                steps_from_which_to_update = [first_step_of_memory]
            else:
                steps_from_which_to_update = []
        else:
            steps_from_which_to_update = range(
                first_step_of_memory, last_step_of_memory + 1
            )

        # We update the Q values
        for step in steps_from_which_to_update:
            transition = self.memory.get_transition(step)
            self.update_from_transition(transition)

        # If we have reached a terminal state, we reset the memory.
        if done:
            self.memory.reset()

        # Log the metrics
        metrics = {}
        metrics.update(
            self.get_metrics_at_transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )
        )
        return metrics

    def update_from_transition(self, transition: Dict[str, Any]) -> None:
        """Update the Q values from the transition"""
        # Unpack the transition
        s_t = transition["state"]
        a_t = transition["action"]
        assert (
            "future_return" in transition
        ), "The transition should have the future_return key"
        g_t = transition["future_return"]

        # Update the Q value
        self.q_values[s_t][a_t] += self.learning_rate.get_value() * (
            g_t - self.q_values[s_t][a_t]
        )
