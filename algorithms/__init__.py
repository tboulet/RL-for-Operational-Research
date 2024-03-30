from typing import Dict, Type, Any, Tuple

from algorithms.base.base_algorithm import BaseRLAlgorithm

from algorithms.random import RandomAlgorithm
from algorithms.q_learning import Q_Learning
from algorithms.q_learning_v0 import Q_learning_v0
from algorithms.double_q_learning import (
    DoubleQ_Learning,
)
from algorithms.reinforce import REINFORCE
from algorithms.sarsa import SARSA
from algorithms.n_step_sarsa import n_step_SARSA
from algorithms.monte_carlo import MonteCarlo


algo_name_to_AlgoClass: Dict[str, Type[BaseRLAlgorithm]] = {
    "Random": RandomAlgorithm,
    "Q-Learning": Q_Learning,
    "Q-Learning v0": Q_learning_v0,
    "Double Q-Learning": DoubleQ_Learning,
    "SARSA": SARSA,
    "n-step SARSA": n_step_SARSA,
    "Monte Carlo": MonteCarlo,
    "REINFORCE": REINFORCE,
}
