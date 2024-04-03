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


class MemoryNSteps:

    def __init__(self, keys: List[str], n_steps: int) -> None:
        """Initialize the memory of the n-steps algorithm.

        Args:
            keys (List[str]): the keys of the transitions that will be stored in the memory
            n_steps (int): the number of steps to store in the memory
        """
        assert n_steps > 0, "The number of steps should be strictly positive"
        # Initialize the parameters
        self.n_steps = n_steps
        self.keys = keys
        # Initialize the memory
        self.steps_in_memory = deque(maxlen=self.n_steps)
        self.step_to_transitions: Dict[str, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.steps_in_memory)

    def is_empty(self) -> bool:
        return len(self) == 0

    def is_full(self) -> bool:
        return len(self) == self.n_steps

    def store_transition(
        self,
        transition: Dict[str, Any],
    ) -> None:
        """Append a transition to the memory. A transition is a dictionnary with keys such as "state", "action", "reward", "next_state" and "done".
        There can also be other keys such as "probs" for the probabilities of the actions if needed by the agent.

        Args:
            transition (Dict[str, Any]): the transition to append to the memory
        """
        assert len(self.steps_in_memory) == len(
            self.step_to_transitions
        ), "The memory and the dictionnary of transitions are not consistent in size"
        # Initialize the memory if it is empty
        if self.is_empty():
            self.steps_in_memory.append(0)
            self.step_to_transitions = {0: transition}

        # Else, append the transition to the memory
        else:
            first_step_of_memory = self.steps_in_memory[0]
            last_step_of_memory = self.steps_in_memory[-1]
            # Remove the first step if the memory is full
            if len(self) == self.n_steps:
                assert (
                    self.steps_in_memory.popleft() == first_step_of_memory
                ), "The first step of the memory is not consistent with the steps in memory"
                del self.step_to_transitions[first_step_of_memory]
            # Add the new transition to the memory
            self.steps_in_memory.append(last_step_of_memory + 1)
            self.step_to_transitions[last_step_of_memory + 1] = transition
            # Assert the two last transitions are consistent
            if "next_state" in self.keys and last_step_of_memory in self.step_to_transitions:
                assert (
                    self.step_to_transitions[last_step_of_memory]["next_state"]
                    == transition["state"]
                ), f"The last next state of the memory is not the current state of the transition, transition : {transition}, last transition : {self.step_to_transitions[last_step_of_memory]}"
            if "done" in self.keys and last_step_of_memory in self.step_to_transitions:
                assert (
                    self.step_to_transitions[last_step_of_memory]["done"] == False
                ), f"We should not append a transition to the memory if the last next state was a terminal state, transition : {transition}, last transition : {self.step_to_transitions[last_step_of_memory]}"

    def get_transition(self, step: int) -> Dict[str, Any]:
        """Get the transition at the given step in the memory.

        Args:
            step (int): the step of the transition to get

        Returns:
            Dict[str, Any]: the transition at the given step
        """
        assert (
            step in self.steps_in_memory
        ), f"The step {step} is not in the memory, which contains steps {self.steps_in_memory}"
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

    def __repr__(self) -> str:
        return f"MemoryNSteps(steps={self.steps_in_memory}, transitions={self.step_to_transitions})"


