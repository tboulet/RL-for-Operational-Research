""" A general algorithm for value based reinforcement learning. It is a paradigm that covers all algorithms that learn a value function (Q or V) and then use this value function to act.

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
from typing import Dict, List, Type, Any, Tuple
import cProfile

# ML libraries
import random
import numpy as np

# File specific
from abc import ABC, abstractmethod
from src.constants import INF
from src.memory import MemoryNSteps
from src.metrics import get_q_values_metrics, get_scheduler_metrics_of_object
from src.schedulers import get_scheduler

# Project imports
from src.typing import QValues, State, Action, StateValues
from src.utils import try_get
from algorithms.base_algorithm import BaseRLAlgorithm


class GeneralizedPolicyIterator(BaseRLAlgorithm):
    """A general algorithm for value based reinforcement learning."""

    keys_all = ["state", "action", "reward", "next_state", "done", "prob"]

    def __init__(
        self,
        config: Dict,
        keys: List[str],
        do_terminal_learning: bool,
        n_steps: int,
        do_compute_returns: bool,
        do_learn_q_values: bool,
        do_learn_states_values: bool,
    ):
        """Initialize the GeneralizedPolicyIterator class.

        Args:
            config (Dict): the config of the algorithm
            keys (List[str]): the keys of the transitions to remember. They must be among ["state", "action", "reward", "next_state", "done"]. Some algorithms don't need the use of certain keys.
            do_terminal_learning (bool): whether to do terminal learning, i.e. to learn at the end of the episode. In that case, if the returns need to be computed, they will be computed at the end of the episode. If not, they will be computed online from the truncated episode.
            n_steps_truncating (int): in case of non-terminal learning, the number of steps to truncate the episode
            do_compute_returns (bool): whether to compute the returns (either online (O(T²) additional operations) or at the end of the episode (O(T) additional operations))
            do_learn_q_values (bool): whether to learn the Q values
            do_learn_states_values (bool): whether to learn the state values
        """
        super().__init__(config=config)
        
        # Numerical hyperparameters
        self.gamma = get_scheduler(config["gamma"])
        self.learning_rate = get_scheduler(config["learning_rate"])
        
        # Keys of the transitions (what to remember in the memory)
        self.keys: List[str] = keys
        assert all(
            [key in self.keys_all for key in self.keys]
        ), f"Keys should be in {self.keys_all}"
        
        # Initialize the memory
        self.do_terminal_learning = do_terminal_learning
        if self.do_terminal_learning:
            self.n_steps = INF
        else:
            self.n_steps = n_steps
        self.memory = MemoryNSteps(keys=keys, n_steps=self.n_steps)
        
        # Whether to compute the returns (online if not terminal O(T^2), or at the end of the episode if terminal (O(T)))
        self.do_compute_returns = do_compute_returns
        if self.do_terminal_learning and not self.do_compute_returns:
            print(
                "WARNING : You are doing terminal learning without computing returns. The point of terminal learning is to efficiently and accurately compute the returns."
            )
        # What to learn
        self.do_learn_q_values = do_learn_q_values
        self.do_learn_states_values = do_learn_states_values

        # Initialize the values
        if do_learn_q_values:
            self.q_values: QValues = self.initialize_q_values(config=config)
        if do_learn_states_values:
            self.state_values: StateValues = self.initialize_state_values(config=config)

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
        transition = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }
        transition = {key: transition[key] for key in transition if key in self.keys}
        if "prob" in self.keys:
            transition["prob"] = self.policy.get_prob(state=state, action=action, is_eval=False)
        self.handle_transition_for_memory(transition=transition)

        first_step_of_memory = self.memory.steps_in_memory[0]
        last_step_of_memory = self.memory.steps_in_memory[-1]

        # Compute the returns/discounted returns if needed.
        if not self.do_terminal_learning and self.do_compute_returns:
            future_return = 0
            for step in range(last_step_of_memory, first_step_of_memory - 1, -1):
                transition = self.memory.get_transition(step)
                reward = transition["reward"]
                future_return = reward + self.gamma.get_value() * future_return
                self.memory.set_transition_key(step, "future_return", future_return)

        # We learn the values
        metrics = {}
        if done or self.memory.is_full():
            sequence_of_transitions = [
                self.memory.get_transition(step)
                for step in range(first_step_of_memory, last_step_of_memory + 1)
            ]
            assert len(sequence_of_transitions) > 0, "The sequence should not be empty"
            metrics = self.update_from_sequence_of_transitions(
                sequence_of_transitions=sequence_of_transitions
            )

        # If we have reached a terminal state, we reset the memory.
        if done:
            self.memory.reset()

        # Log the metrics
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

    @abstractmethod
    def update_from_sequence_of_transitions(
        self, sequence_of_transitions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """The update of the values from an sequence of transitions. This method is specific to each algorithm.
        The transition could be enriched with the returns, the advantage, the probs, etc.
        Also returns eventual metrics.

        Args:
            sequence_of_transitions (List[Dict[str, Any]]): the sequence of transitions to learn from

        Returns:
            Dict[str, float]: the metrics computed during the update
        """

    # ============ Helper methods ============

    def handle_transition_for_memory(self, transition: Dict[str, Any]) -> None:
        """Store an transition in the memory, and if we have reached a terminal state, reset the memory and store the n-step transition.
        This method has to be called at the end of the update method of the algorithm.

        Args:
            transition (Dict[str, Any]): the transition to store
        """
        # Store the transition in the memory
        self.memory.store_transition(transition)
        # Compute the returns/discounted returns if needed. This operation is in O(t), which makes the memory management O(t²)
        if self.do_compute_returns:
            first_step_of_memory = self.memory.steps_in_memory[0]
            last_step_of_memory = self.memory.steps_in_memory[-1]
            reward = transition["reward"]
            self.memory.set_transition_key(
                last_step_of_memory, "future_return", reward
            )
            reward_discounted = reward * self.gamma.get_value()
            if not self.memory.is_empty():
                for step in range(
                    last_step_of_memory - 1, first_step_of_memory - 1, -1
                ):
                    self.memory.set_transition_key(
                        step,
                        "future_return",
                        reward_discounted
                        + self.memory.get_transition_key(step, "future_return"),
                    )
                    reward_discounted *= self.gamma.get_value()
                    
                    
    def compute_n_step_sarsa_target(
        self,
        sequence_of_transitions: List[Dict[str, Any]],
        n_steps_for_sarsa: int,
    ) -> float:
        """Compute the n-step SARSA target from a sequence of transitions.

        Args:
            sequence_of_transitions (List[Dict[str, Any]]): the sequence of transitions
            gamma (Scheduler): the scheduler of the discount factor gamma
            n_steps (int): the number of steps of the SARSA algorithm

        Returns:
            float: the n-step SARSA target
        """
        assert (
            len(sequence_of_transitions) > 0
        ), f"The sequence of transitions should not be empty"
        assert "done" in self.keys, f"The key 'done' should be in the keys"
        gamma = self.gamma.get_value()
        target = 0
        k = 0
        while True:
            if sequence_of_transitions[k]["done"]:
                # If we are at the end of the episode, we add the reward and we stop
                target += (gamma**k) * sequence_of_transitions[k]["reward"]
                break
            elif k < n_steps_for_sarsa - 1:
                # If we are not at the end of the episode, and not at the penultimate transition, we add the dizscounted reward and we continue
                target += (gamma**k) * sequence_of_transitions[k]["reward"]
            elif k == n_steps_for_sarsa - 1:
                # If we are at the penultimate transition, we add the discounted reward and the Q value of the next state and action, then we stop
                reward = sequence_of_transitions[k]["reward"]
                if not sequence_of_transitions[k + 1]["done"]:
                    next_state = sequence_of_transitions[k + 1]["state"]
                    next_action = sequence_of_transitions[k + 1]["action"]
                    target += (gamma**k) * (
                        reward + gamma * self.q_values[next_state][next_action]
                    )
                else:
                    target += (gamma**k) * reward
                break
            k += 1
        return target
