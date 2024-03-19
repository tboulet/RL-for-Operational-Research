from typing import Dict, List, Type, Any, Tuple

import numpy as np
from algorithms.base.base_algorithm import BaseRLAlgorithm
from src.typing import Action, State


class RandomAlgorithm(BaseRLAlgorithm):
    """A random agent that selects actions uniformly at random."""

    def __init__(self, config: Dict):
        super().__init__(config)

    def act(self, state: State, available_actions: List[Action], is_eval: bool = False) -> Action:
        """Perform a random action.

        Args:
            state (State): the current state (or observation)
            available_actions (List[Action]): the list of available actions for the agent to choose from
            is_eval (bool, optional): whether the agent is evaluating or not. Defaults to False.

        Returns:
            Action: the action to perform according to the agent
        """
        return np.random.choice(available_actions)

    def update(self, state: State, action: Action, reward: float, next_state: State, done: bool) -> None:
        pass