from collections import defaultdict
from typing import Dict, List, Type, Any, Tuple

import numpy as np
from algorithms.base_algorithm import BaseRLAlgorithm
from src.constants import EPSILON
from src.typing import Action, State


class MonteCarloTreeSearchAlgorithm(BaseRLAlgorithm):
    """A Monte Carlo Tree Search agent."""

    def __init__(self, config: Dict):
        # Set the hyperparameters
        self.c = config["c"]
        self.gamma = config["gamma"]
        self.update_method = config["update_method"]
        self.learning_rate = config["learning_rate"]
        self.decay_lr = config["decay_lr"]
        # Initialize the Q values and the N_seen counts
        self.Q_values: Dict[Tuple[State, Action], float] = defaultdict(lambda: 0)
        self.N_seen : Dict[Tuple[State, Action], int] = defaultdict(lambda: 0)
        self.episodic_memory : List[Tuple[State, Action, float]] = []
        super().__init__(config)

    def act(
        self, state: State, available_actions: List[Action], is_eval: bool = False
    ) -> Action:
        """Perform an action based on the UCB formula, which maximize a tradeoff between exploration and exploitation.
        The exploitation is the average reward of the action.
        The exploration is a term that explodes if the action has not been tried often in that state.

        Args:
            state (State): the current state (or observation)
            available_actions (List[Action]): the list of available actions for the agent to choose from
            is_eval (bool, optional): whether the agent is evaluating or not. Defaults to False.

        Returns:
            Action: the action to perform according to the agent
        """
        # Pick best action according to UCB formula
        best_action = None
        best_ucb_value = -np.inf
        for action in available_actions:
            # Play the action if we have never seen it in that state
            if self.N_seen[(state, action)] == 0:
                return action
            # Compute the UCB value
            ucb_value = self.Q_values[(state, action)] + self.c * np.sqrt(
                np.log(self.N_seen[state]) / self.N_seen[(state, action)]
            )
            # Update the best action if we have found a better one
            if ucb_value > best_ucb_value:
                best_ucb_value = ucb_value
                best_action = action
        action = best_action
        assert len(available_actions) != 0, "No available actions"
        assert action is not None, "No action was selected"
                
        # Return the action
        return action

    def update(
        self, state: State, action: Action, reward: float, next_state: State, done: bool
    ) -> None:
        
        # Increment the number of times we have seen the state and chosen the state-action pair
        self.N_seen[state] += 1
        self.N_seen[(state, action)] += 1
        
        # Add the reward to the rollout history
        self.episodic_memory.append((state, action, reward))
        
        # If done, backpropagate the reward to any state-action pair we have seen
        if done:
            T = len(self.episodic_memory)
            Gt = 0
            for i in range(T):
                # Work backwards through the rollout history, with t being the time step
                t = T-i-1
                # Get the (s, a, r) tuple from the rollout history
                s, a, r = self.episodic_memory[t]
                # Compute Gt
                Gt = r + self.gamma * Gt
                # Update the Q value
                if self.update_method == "average":
                    self.Q_values[(s, a)] += (1/self.N_seen[(s, a)]) * (Gt - self.Q_values[(s, a)])
                elif self.update_method == "learning_rate":
                    self.Q_values[(s, a)] += self.learning_rate * (Gt - self.Q_values[(s, a)])
                else:
                    raise ValueError(f"Unknown update method {self.update_method}")
        
            # Reset the rollout history
            self.episodic_memory = []
            
            # Decay the learning rate
            self.learning_rate *= self.decay_lr
