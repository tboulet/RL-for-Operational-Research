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
from src.schedulers import Scheduler
from src.typing import Action, State

# Project imports


class BaseLearner(ABC):
    """Base class for the learners object.
    Learners are python objects that learn a target function from examples iteratively.

    Each learning step of a learner L : *args -> E (E in P(R^n)) is under the form :
    L(*conditional variables) <- X

    Which represents :
    L(*conditional variables) tending (according to the learning algorithm Alg(L)) to E[X|*conditional variables]

    Where :
    - X is the target, as a scalar or a vector : shape = (n,)
    - *conditional variables are the variables that condition the learning process, in an observation format : shape = (len(*conditional variables), n, *dim_variable)
    - Alg(L) is the learning algorithm of the learner L, e.g. gradient descent, cumulative average, moving average, etc.
    """

    def __init__(self, learning_rate: Scheduler) -> None:
        self.learning_rate = learning_rate

    @abstractmethod
    def learn(
        self,
        *conditional_variables: List[Any],
        target: Union[float, int, np.ndarray],
        multiplier_error : Optional[float] = 1,
    ) -> Dict[str, float]:
        """Learn from the target and the conditional variables.

        Args:
            *conditional_variables (List[Any]): The variables that condition the learning process, of shape (len(*conditional variables), n, *dim_variable).
            target (Union[float, int, np.ndarray]): The target to learn, of shape (n,).
            multiplier_error (Optional[float], optional): The multiplier of the error. Defaults to 1.
            
        Returns:
            Dict[str, float]: The metrics of the learning process.
        """
        pass

    @abstractmethod
    def __call__(
        self, *conditional_variables: List[Any]
    ) -> Union[float, int, np.ndarray]:
        """Predict the target from the conditional variables.

        Args:
            *conditional_variables (List[Any]): The variables that condition the learning process, of shape (len(*conditional variables), n, *dim_variable).

        Returns:
            Union[float, int, np.ndarray]: The predicted target, of shape (n,).
        """
        pass


class BaseQValuesLearner(BaseLearner):
    """The base class for any Q value learner.
    This class is an interface for forcing the get_availble_actions method.
    """

    @abstractmethod
    def get_available_actions(self, state: State) -> List[Action]:
        """Return the available actions for the given state.

        Args:
            state (State): the state for which to get the available actions

        Returns:
            List[Action]: the available actions
        """
        pass

    @abstractmethod
    def __call__(
        self, state: State, action: Optional[Action] = None
    ) -> Union[float, int, np.ndarray, Dict[Action, Union[float, int, np.ndarray]]]:
        """Return the Q values for the given conditional variables.

        Args:
            state (State): the state for which to get the Q values
            action (Optional[Action], optional): the action for which to get the Q value. Defaults to None.

        Returns:
            Union[float, int, np.ndarray, Dict[Action, Union[float, int, np.ndarray]]]: the Q values
        """
