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

# File specific
from abc import ABC, abstractmethod

from src.typing import Action, QValues
from src.utils import instantiate_class


class Scheduler(ABC):
    """The base class for any scheduler. A scheduler is a class that can return a value at each step that
    depends on the step number.

    The step number has to be updated using the "update_step" method of the scheduler.
    """

    def __init__(
        self,
        step_init: int = 0,
        upper_bound: float = None,
        lower_bound: float = None,
    ):
        """Initializes the scheduler

        Args:
            step_init (int, optional): the initial step number of the scheduler, usually 0. Defaults to 0.
            upper_bound (float, optional): the upper bound of the scheduler's return. Defaults to None.
            lower_bound (float, optional): the lower bound of the scheduler's return. Defaults to None.
        """
        self.step = step_init
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def update_step(self, new_step: int):
        """Update the step number of the scheduler.

        Args:
            new_step (int): the new step number to set
        """
        self.step = new_step

    def get_step(self) -> int:
        """Return the current step number of the scheduler.

        Returns:
            int: the current step number
        """
        return self.step

    def increment_step(self):
        """Increment the step number of the scheduler by 1."""
        self.step += 1

    @abstractmethod
    def _get_value(self, step: int) -> float:
        """The method that should be implemented by the child class to return the value at each step

        Args:
            step (int): the current step

        Returns:
            float: the value at the current step
        """

    def get_value(self) -> float:
        """Return a value at the current step. If the value is outside the bounds, it will be bounded.

        Returns:
            float: the value at the current step
        """
        current_step = self.step
        res = self._get_value(step=current_step)
        if self.upper_bound is not None:
            assert isinstance(
                res, (int, float)
            ), f"Result returned by the scheduler is not a number : {res}, can't bound it."
            res = min(self.upper_bound, res)
        if self.lower_bound is not None:
            res = max(self.lower_bound, res)
            assert isinstance(
                res, (int, float)
            ), f"Result returned by the scheduler is not a number : {res}, can't bound it."
        return res


class Constant(Scheduler):
    """A scheduler that returns a constant value at each step."""

    def __init__(self, value: float, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def _get_value(self, step: int) -> float:
        return self.value


class Linear(Scheduler):
    """A scheduler that returns a linearly increasing/decreasing value at each step."""

    def __init__(self, start_value: float, end_value: float, n_steps: int, **kwargs):
        """Initializes the Linear scheduler.
        It is parameterized by the value at the first step and the value at the n_steps-th step.
        Aftert that n-steps, the value will continue to follow the linear behavior.

        Args:
            start_value (float): the value at the first step
            end_value (float): the value at the n_steps-th step
            n_steps (int): the number of steps after which the value should be end_value.
        """
        super().__init__(**kwargs)
        self.start_value = start_value
        self.end_value = end_value
        self.n_steps = n_steps

    def _get_value(self, step: int) -> float:
        if step > self.n_steps:
            return self.end_value
        else:
            return (
                self.start_value
                + (self.end_value - self.start_value) * step / self.n_steps
            )


class Exponential(Scheduler):
    """A scheduler that returns an exponentially increasing/decreasing value at each step."""

    def __init__(
        self,
        start_value: float,
        end_value: float,
        n_steps: int,
        **kwargs,
    ):
        """Initializes the Exponential scheduler.
        It is parameterized by the value at the first step and the value at the n_steps-th step.
        Aftert that n-steps, the value will continue to follow the exponential behavior.

        Args:
            start_value (float): the value at the first step
            end_value (float): the value at the n_steps-th step
            n_steps (int): the number of steps after which the value should be end_value.
        """
        super().__init__(**kwargs)
        self.start_value = start_value
        self.end_value = end_value
        self.n_steps = n_steps

    def _get_value(self, step: int) -> float:
        return self.start_value * (self.end_value / self.start_value) ** (
            step / self.n_steps
        )


class Inverse(Scheduler):
    """A scheduler that returns the inverse of the step number at each step."""

    def __init__(
        self,
        value_start: float,
        value_target: float,
        value_n_steps: Optional[float] = None,
        n_steps: int = None,
        **kwargs,
    ):
        """Initializes the InverseScheduler scheduler.
        
        Args:
            value_start (float): the value at the first step
            value_target (float): the value towards which the scheduler should converge
            value_n_steps (int): the value at the n_steps-th step, if unspecified, the scheduler will pick a convergence rate of 1
            n_steps (int): the number of steps after which the value should be value_n_steps.
        """
        super().__init__(**kwargs)
        if value_n_steps is None:
            self.convergence_rate = 1
        else:
            assert n_steps is not None and n_steps > 0, "If value_n_steps is specified, n_steps should be specified too as a positive integer"
            assert (value_start == value_n_steps == value_target) or (value_start < value_n_steps < value_target) or (value_start > value_n_steps > value_target), "The values 'value_start', 'value_n_steps' and 'value_target' should be in increasing or decreasing order"
            if value_n_steps == value_target: # constant value at value_target
                self.convergence_rate = 1
            else:
                self.convergence_rate = ((value_start - value_target) / (value_n_steps - value_target) - 1) / n_steps
        self.start_value = value_start
        self.target_value = value_target
        
    def _get_value(self, step: int) -> float:
        return self.target_value + (self.start_value - self.target_value) / (1 + self.convergence_rate * step)
    
    
class SquareWave(Scheduler):
    """A square wave scheduler, that alternates between two values at each step."""

    def __init__(
        self,
        max_value: int,
        min_value: int,
        steps_at_min: int,
        steps_at_max: int,
        start_at_max: bool,
        **kwargs,
    ):
        """Initializes the SquareWave scheduler.
        It is parameterized by the maximum value, the minimum value, the number of steps at the minimum value and the number of steps at the maximum value.
        There is also a parameter to choose if the scheduler should start at the maximum value or not.

        Args:
            max_value (int): the maximum value of the square wave
            min_value (int): the minimum value of the square wave
            steps_at_min (int): the number of steps at the minimum value for each period
            steps_at_max (int): the number of steps at the maximum value for each period
            start_at_max (bool): whether to start at the maximum value or not
        """
        super().__init__(**kwargs)
        self.max_value = max_value
        self.min_value = min_value
        self.steps_at_min = steps_at_min
        self.steps_at_max = steps_at_max
        self.start_at_max = start_at_max

    def _get_value(self, step: int) -> float:
        step = step % (self.steps_at_max + self.steps_at_min)
        if self.start_at_max:
            if step < self.steps_at_max:
                return self.max_value
            else:
                return self.min_value
        else:
            if step < self.steps_at_min:
                return self.min_value
            else:
                return self.max_value


def get_scheduler(config_or_value: Union[Dict, float, int]) -> Scheduler:
    """Return a scheduler from either a configuration or a value.

    If it's a simple numerical value, a Constant scheduler will be returned.

    If it's a configuration, the scheduler will be initialized based on the configuration dictionary.
    This dictionnary should contain a key "class_string" of the form "path.to.module:ClassName"
    and other parameters that will be passed to the scheduler's constructor.

    Args:
        config_or_value (Union[Dict, float, int]): the configuration or value of the scheduler

    Returns:
        Scheduler: the scheduler
    """
    if isinstance(config_or_value, (int, float)):
        return Constant(value=config_or_value)
    else:
        return instantiate_class(config_or_value)
