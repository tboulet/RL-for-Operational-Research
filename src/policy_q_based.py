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
from src.constants import EPSILON
from src.schedulers import Scheduler, get_scheduler

from src.typing import Action, QValues, State, QValues


class PolicyQBased(ABC):
    """The base interface for all Q-based policies. This class
    should be subclassed when implementing a new policy based on Q values.

    This class requires to implement the "get_probabilities" method, and
    optionally to subclass the "act" method (for instance to update internal state).
    """

    def __init__(
        self,
        q_values: QValues,
    ):
        """Initialize the policy with the given configuration.

        Args:
            q_values (QValues): the Q values of the policy
        """
        self.q_values = q_values

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
        probabilities = self.get_probabilities(
            state=state,
            available_actions=list(self.q_values[state].keys()),
            is_eval=is_eval,
        )
        return probabilities[action]

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

    def get_greedy_action(
        self,
        state: State,
        available_actions: List[Action],
    ) -> Action:
        """Return the greedy action for a given state.

        Args:
            state (State): the state to get the greedy action for
            available_actions (List[Action]): the list of available actions for the agent to choose from

        Returns:
            Action: the greedy action for the given state
        """
        assert (
            len(available_actions) > 0
        ), "There should be at least one available action"
        return max(available_actions, key=lambda a: self.q_values[state][a])


class PolicyGreedy(PolicyQBased):
    """The greedy policy for Q-learning. It always picks the action with the highest Q-value."""

    def get_probabilities(
        self,
        state: State,
        available_actions: List[Action],
        is_eval: bool,
    ) -> Dict[Action, float]:
        greedy_action = self.get_greedy_action(
            state=state, available_actions=available_actions
        )
        return {a: 1 if a == greedy_action else 0 for a in self.q_values[state]}


class PolicyEpsilonGreedy(PolicyQBased):

    def __init__(
        self,
        q_values: QValues,
        epsilon: Union[float, int, Scheduler],
    ):
        """The epsilon-greedy policy for Q-learning.

        Args:
            q_values (QValues): the Q values of the policy
            epsilon (Union[float, int, Scheduler]): the epsilon value or scheduler config.
        """
        super().__init__(q_values=q_values)
        self.epsilon: Scheduler = get_scheduler(config_or_value=epsilon)

    def get_probabilities(
        self,
        state: State,
        available_actions: List[Action],
        is_eval: bool,
    ) -> Dict[Action, float]:
        greedy_action = self.get_greedy_action(
            state=state, available_actions=available_actions
        )
        if is_eval:
            # In eval mode, this is the greedy policy
            return {a: 1 if a == greedy_action else 0 for a in self.q_values[state]}
        else:
            # In training mode, we use epsilon-greedy
            n_actions = len(self.q_values[state])
            eps = self.epsilon.get_value()
            probabilities = {
                a: 1 - eps + eps / n_actions if a == greedy_action else eps / n_actions
                for a in self.q_values[state]
            }
            # Return the probabilities
            return probabilities

    def act(
        self,
        state: Any,
        available_actions: List[Action],
        is_eval: bool = False,
    ) -> Action:

        if is_eval:
            # In eval mode, this is the greedy policy
            return (
                self.get_greedy_action(
                    state=state, available_actions=available_actions
                ),
                1,
            )

        else:
            # In train mode, we use epsilon-greedy
            probabilities = self.get_probabilities(
                state=state, available_actions=available_actions, is_eval=is_eval
            )
            action = np.random.choice(
                list(probabilities.keys()), p=list(probabilities.values())
            )
            prob = probabilities[action]

            # Update the epsilon scheduler
            self.epsilon.increment_step()

            return action, prob


class PolicyBoltzmann(PolicyQBased):

    def __init__(self, q_values: QValues, temperature: Union[float, int, Scheduler]):
        """The Boltzmann policy for Q-learning. It assigns probabilities to each action proportional to the exponentiated Q-value.
        The temperature parameter controls the randomness of the policy. A high temperature leads to a more random policy, while a low temperature leads to a more deterministic policy.

        Args:
            q_values (QValues): the Q values of the policy
            temperature (Union[float, int, Scheduler]): the temperature value or scheduler config.
        """
        super().__init__(q_values=q_values)
        self.temperature: Scheduler = get_scheduler(config_or_value=temperature)

    def get_probabilities(
        self,
        state: State,
        available_actions: List[Action],
        is_eval: bool,
    ) -> Dict[Action, float]:
        assert (
            len(available_actions) > 0
        ), "There should be at least one available action"
        if is_eval:
            # In eval mode, this is the greedy policy
            return {
                a: (
                    1
                    if a
                    == self.get_greedy_action(
                        state=state,
                        available_actions=available_actions,
                    )
                    else 0
                )
                for a in self.q_values[state]
            }
        else:
            # In training mode, we use the Boltzmann policy
            temperature = self.temperature.get_value()
            q_values_at_state = np.array(
                [self.q_values[state][a] for a in available_actions]
            )
            probabilities = np.exp(q_values_at_state / temperature) / np.sum(
                np.exp(q_values_at_state / temperature)
            )
            # Return the probabilities
            return {
                available_actions[i]: probabilities[i]
                for i in range(len(available_actions))
            }

    def act(
        self,
        state: Any,
        available_actions: List[Action],
        is_eval: bool = False,
    ) -> Action:
        assert (
            len(available_actions) > 0
        ), "There should be at least one available action"
        if is_eval:
            # In eval mode, this is the greedy policy
            return (
                self.get_greedy_action(
                    state=state, available_actions=available_actions
                ),
                1,
            )
        else:
            # In train mode, we use the Boltzmann policy
            probabilities = self.get_probabilities(
                state=state, available_actions=available_actions, is_eval=is_eval
            )
            action = np.random.choice(
                list(probabilities.keys()), p=list(probabilities.values())
            )
            prob = probabilities[action]

            # Update the temperature scheduler
            self.temperature.increment_step()

            return action, prob


class PolicyUCB(PolicyQBased):

    def __init__(self, q_values: QValues, ucb_constant: Union[float, int, Scheduler]):
        """The Upper Confidence Bound (UCB) policy for Q-learning.
        It pick the action that maximizes a trade-off between exploitation and exploration.
        The exploitation term is the Q-value, while the exploration term is a term that explodes when the action has not been tried often in that state.

        Args:
            q_values (QValues): the Q values of the policy
            ucb_constant (Union[float, int, Scheduler]): the exploration bonus value or scheduler config.
        """
        super().__init__(q_values=q_values)
        self.ucb_constant: Scheduler = get_scheduler(config_or_value=ucb_constant)
        self.n_seen_observed = defaultdict(lambda: defaultdict(int))

    def get_best_ucb_action(
        self,
        state: State,
        available_actions: List[Action],
    ) -> Action:
        """Get the action that maximizes the UCB value for a given state.

        Args:
            state (State): the state to get the action for
            available_actions (List[Action]): the list of available actions for the agent to choose from

        Returns:
            Action: the action that maximizes the UCB value
        """
        ucb_constant_value = self.ucb_constant.get_value()
        n_total = sum(self.n_seen_observed[state].values())
        ucb_values = {
            a: self.q_values[state][a]
            + ucb_constant_value
            * np.sqrt(np.log(n_total + 1) / (EPSILON + self.n_seen_observed[state][a]))
            for a in available_actions
        }
        best_action = max(ucb_values, key=ucb_values.get)
        return best_action

    def get_probabilities(
        self,
        state: State,
        available_actions: List[Action],
        is_eval: bool,
    ) -> Dict[Action, float]:
        assert (
            len(available_actions) > 0
        ), "There should be at least one available action"
        if is_eval:
            # In eval mode, this is the greedy policy
            greedy_action = self.get_greedy_action(
                state=state, available_actions=available_actions
            )
            return {a: 1 if a == greedy_action else 0 for a in available_actions}
        else:
            # In training mode, we use the UCB policy
            best_ucb_action = self.get_best_ucb_action(
                state=state,
                available_actions=available_actions,
            )
            # Return the probabilities
            return {a: 1 if a == best_ucb_action else 0 for a in available_actions}

    def act(
        self,
        state: Any,
        available_actions: List[Action],
        is_eval: bool = False,
    ) -> Action:
        assert (
            len(available_actions) > 0
        ), "There should be at least one available action"
        if is_eval:
            # In eval mode, this is the greedy policy
            greedy_action = self.get_greedy_action(
                state=state, available_actions=available_actions
            )
            return greedy_action, 1
        else:
            # In train mode, we use the UCB policy
            best_ucb_action = self.get_best_ucb_action(
                state=state, available_actions=available_actions
            )
            # Update the number of times the action has been tried
            self.n_seen_observed[state][best_ucb_action] += 1
            # Update the UCB constant scheduler
            self.ucb_constant.increment_step()
            return best_ucb_action, 1
