from typing import Dict, Type, Any, Tuple
from algorithms.SARSA import SARSA_v0
from algorithms.base_algorithm import BaseRLAlgorithm
from algorithms.general_policy_iterator_algorithms.double_q_learning import (
    DoubleQ_Learning,
)
from algorithms.general_policy_iterator_algorithms.monte_carlo import MonteCarlo
from algorithms.general_policy_iterator_algorithms.n_step_sarsa import n_step_SARSA
from algorithms.general_policy_iterator_algorithms.q_learning import Q_Learning
from algorithms.general_policy_iterator_algorithms.sarsa import SARSA
from algorithms.random import RandomAlgorithm
from algorithms.r2 import Random2
from algorithms.Q_learning import Q_learning_v0


algo_name_to_AlgoClass: Dict[str, Type[BaseRLAlgorithm]] = {
    "Random": RandomAlgorithm,
    "Random2": Random2,
    "Q-Learning v0": Q_learning_v0,
    "Double Q-Learning": DoubleQ_Learning,
    "SARSA v0": SARSA_v0,
    "Monte Carlo": MonteCarlo,
    "SARSA": SARSA,
    "n-step SARSA": n_step_SARSA,
    "Q-Learning": Q_Learning,
}
