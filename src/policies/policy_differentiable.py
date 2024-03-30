# Logging
from abc import abstractmethod
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
from src.policies.base_policy import BasePolicy
from src.typing import Action, QValues, State
import cProfile

# ML libraries
import random
import numpy as np

from src.utils import get_softmax_probs


class PolicyDifferentiable(BasePolicy):
    """An interface class that requires to implement the method `gradient_step` which will modify the policy's parameters by adding the gradient."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def gradient_step(self, **kwargs) -> None:
        """Modify the policy's parameters by adding the gradient.

        Args:
            **kwargs: the arguments to pass to the gradient step
        """
        pass


class PolicyTabular(PolicyDifferentiable):
    """A tabular policy that stores the parameters as logits in a table.

    In the tabular case, the gradient ascent at t corresponds to the following formula :
    For each action a in A(s_t) :
        logit += learning_rate * (1(a==a_t) - prob(a|s_t))
    """

    def __init__(self) -> None:
        super().__init__()
        self.logits: Dict[State, Dict[Action, float]] = defaultdict(
            lambda: defaultdict(lambda: np.random.normal(0, 1))
        )

    def get_probabilities(
        self,
        state: State,
        available_actions: List[Action],
        is_eval: bool,
    ) -> Dict[Action, float]:
        # Initialize the logits if not done
        for a in available_actions:
            self.logits[state][a]
        # Return the probabilities
        return get_softmax_probs(self.logits[state])

    def get_prob(self, state: State, action: Action) -> float:
        return get_softmax_probs(self.logits[state])[action]

    def gradient_step(
        self,
        state: State,
        action: Action,
        learning_rate: float,
        advantage: float,
    ) -> None:
        for a in self.logits[state]:
            self.logits[state][a] += (
                learning_rate
                * (int(a == action) - self.get_prob(state, a))
                * advantage
            )
