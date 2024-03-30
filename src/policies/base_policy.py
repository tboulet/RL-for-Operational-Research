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
from src.typing import Action, QValues, State
import cProfile

# ML libraries
import random
import numpy as np

# File specific
from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """The base interface for all policies.

    This class requires to implement the following methods :
    - get_probabilities : return the probabilities of the actions being chosen for a given state
    - get_prob : return the probability of an action being chosen for a given state

    Optionally, you can subclass the "act" method (for instance to update internal state).
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_probabilities(
        self,
        state: State,
        available_actions: List[Action],
        is_eval: bool,
    ) -> Dict[Action, float]:
        """Return the probabilities of the actions being chosen for a given state.

        Args:
            state (State): the state to get the probabilities for
            available_actions (List[Action]): the list of available actions for the agent to choose from
            is_eval (bool): whether the agent is evaluating or not

        Returns:
            Dict[Action, float]: the probabilities of the actions being chosen
        """

    @abstractmethod
    def get_prob(
        self,
        state: State,
        action: Action,
        is_eval: bool,
    ) -> float:
        """Return the probability of an action being chosen for a given state.

        Args:
            state (State): the state to get the probability for
            action (Action): the action to get the probability for
            is_eval (bool): whether the agent is evaluating or not

        Returns:
            float: the probability of the action being chosen
        """

    def act(
        self,
        state: State,
        available_actions: List[Action],
        is_eval: bool = False,
    ) -> Tuple[Action, Optional[float]]:
        """Perform an action based on the state. Also return the probability of the action being chosen if possible, or None if not.
        This should be interpreted as an exlorative/exploitative training step performed by the agent,
        and should therefore update the policy's internal state (schedulers, number of times actions have been tried, etc.)

        Args:
            state (Any): the current state (or observation)
            available_actions (List[Action]): the list of available actions for the agent to choose from
            is_eval (bool, optional): whether the agent is evaluating or not. Defaults to False.

        Returns:
            Action: the action to perform according to the agent
            Optional[float]: the probability of the action being chosen, or None if not available
        """
        probabilities = self.get_probabilities(
            state=state, available_actions=available_actions, is_eval=is_eval
        )
        action = np.random.choice(
            list(probabilities.keys()), p=list(probabilities.values())
        )
        prob = probabilities[action]
        return action, prob
