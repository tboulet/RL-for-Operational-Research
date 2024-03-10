"""Util class for algorithm that learns from a n-step transition.

Such algorithms include :
    - (n-1)-step SARSA
    - n-step Q-learning
    - n-step Expected SARSA
    - n-step Sampled Expected SARSA
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
from typing import Dict, List, Optional, Type, Any, Tuple, Union
import cProfile

# ML libraries
import random
import numpy as np
from collections import deque

from algorithms.base_algorithm import BaseRLAlgorithm
from src.typing import Action, State


class MemoryNSteps:

    def __init__(self, keys: List[str], n_steps: int) -> None:
        """Initialize the memory of the n-steps algorithm.

        Args:
            keys (List[str]): the keys of the transitions that will be stored in the memory
            n_steps (int): the number of steps to store in the memory
        """
        # Initialize the parameters
        self.n_steps = n_steps
        self.keys = keys
        # Initialize the memory
        self.steps_in_memory = deque(maxlen=self.n_steps)
        self.step_to_transitions: Dict[str, Dict[str, Any]] = {}
        self.first_step = None
        self.last_step = None

    def __len__(self) -> int:
        return len(self.steps_in_memory)

    def is_empty(self) -> bool:
        return len(self) == 0

    def store_transition(
        self,
        transition: Dict[str, Any],
    ) -> None:
        """Append a transition to the memory. A transition is a dictionnary with keys such as "state", "action", "reward", "next_state" and "done".
        There can also be other keys such as "probs" for the probabilities of the actions if needed by the agent.

        Args:
            transition (Dict[str, Any]): the transition to append to the memory
        """

        # Remove the first step if the memory is full
        if len(self.steps_in_memory) == self.n_steps:
            removed_step = self.steps_in_memory.popleft()
            del self.step_to_transitions[removed_step]
            self.first_step = 0 if self.is_empty() else self.first_step + 1
        # Add the new step
        new_step = len(self.steps_in_memory)
        self.steps_in_memory.append(new_step)
        self.step_to_transitions[new_step] = transition
        self.last_step = new_step
        # Assert the two last transitions are consistent
        if len(self.steps_in_memory) > 1:
            assert (
                self.step_to_transitions[self.last_step - 1]["next_state"]
                == transition["state"]
            ), "The last next state of the memory is not the current state of the transition"
            assert (
                self.step_to_transitions[self.last_step - 1]["done"] == False
            ), "We should not append a transition to the memory if the last next state was a terminal state"

    def get_transition(self, step: int) -> Dict[str, Any]:
        """Get the transition at the given step in the memory.

        Args:
            step (int): the step of the transition to get

        Returns:
            Dict[str, Any]: the transition at the given step
        """
        assert (
            step in self.steps_in_memory
        ), f"The step {step} is not in the memory, which goes from {self.first_step} to {self.last_step} (inclusive)"
        return self.step_to_transitions[step]

    def get_transition_key(self, step: int, key: str) -> Any:
        """Get the value of the given key in the transition at the given step in the memory.

        Args:
            step (int): the step of the transition to get
            key (str): the key of the value to get

        Returns:
            Any: the value of the given key in the transition at the given step
        """
        transition = self.get_transition(step)
        assert key in transition, f"The key {key} is not in the transition {transition}"
        return transition[key]

    def set_transition(self, step: int, transition: Dict[str, Any]) -> None:
        """Set the transition at the given step in the memory.

        Args:
            step (int): the step of the transition to get
            transition (Dict[str, Any]): the transition to set at the given step
        """
        self.step_to_transitions[step] = transition

    def set_transition_key(self, step: int, key: str, value: Any) -> None:
        """Set the value of the given key in the transition at the given step in the memory.

        Args:
            step (int): the step of the transition to get
            key (str): the key of the value to get
            value (Any): the value to set for the given key in the transition at the given step
        """
        transition = self.get_transition(step)
        transition[key] = value
        self.set_transition(step, transition)

    def reset(self) -> None:
        self.steps_in_memory = deque(maxlen=self.n_steps)
        self.step_to_transitions = {}
        self.first_step = None
        self.last_step = None


class AlgorithmNSteps(BaseRLAlgorithm):
    """Util class for algorithm that learns from a n-step transition."""

    def __init__(
        self,
        config: Dict,
        keys: List[str],
        n_steps: int,
        do_compute_returns: bool = True,
        do_compute_discounted_returns: bool = True,
        gamma: float = None,
    ) -> None:
        """Initialize the AlgorithmNSteps class. It is parameterized by the number of steps and the keys of the transitions that will be stored in the memory.

        Args:
            config (Dict): the configuration of the algorithm
            n_steps (int): the number of steps to store in the memory
            keys (List[str]): the keys of the transitions that will be stored in the memory
        """
        super().__init__(config)

        self.keys = keys
        self.n_steps = n_steps
        self.do_compute_returns = do_compute_returns
        self.do_compute_discounted_returns = do_compute_discounted_returns
        self.gamma = gamma
        if self.do_compute_discounted_returns:
            assert (
                self.gamma is not None
            ), "If we want to compute discounted returns, we need to have a discount factor"
            assert (
                isinstance(self.gamma, (int, float)) and 0 <= self.gamma <= 1
            ), "The discount factor should be a float between 0 and 1"
        self.memory = MemoryNSteps(keys=keys, n_steps=n_steps)

    def handle_transition_for_memory(self, transition: Dict[str, Any]) -> None:
        """Store an transition in the memory, and if we have reached a terminal state, reset the memory and store the n-step transition.
        This method has to be called at the end of the update method of the algorithm.

        Args:
            transition (Dict[str, Any]): the transition to store
        """
        # Empty the memory if we have reached a terminal state
        if transition["done"]:
            self.memory.reset()
        else:
            # Store the transition in the memory
            self.memory.store_transition(transition)

            # Compute the returns/discounted returns if needed
            if self.do_compute_returns:
                reward = transition["reward"]
                self.memory.set_transition_key(self.memory.last_step, "returns", reward)
                for step in range(self.memory.first_step, self.memory.last_step):
                    self.memory.set_transition_key(
                        step,
                        "returns",
                        reward + self.memory.get_transition_key(step, "returns"),
                    )
            if self.do_compute_discounted_returns:
                reward = transition["reward"]
                self.memory.set_transition_key(
                    self.memory.last_step, "discounted_returns", reward
                )
                reward_discounted = reward
                for step in range(self.memory.last_step - 1, self.memory.first_step - 1, -1):
                    self.memory.set_transition_key(
                        step,
                        "discounted_returns",
                        reward + self.memory.get_transition_key(step, "discounted_returns"),
                    )
                    reward_discounted *= self.gamma
