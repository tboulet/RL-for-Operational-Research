from typing import Dict, Type, Any, Tuple
from algorithms.SARSA import SARSA
from algorithms.base_algorithm import BaseRLAlgorithm
from algorithms.general_policy_iterator_algorithms.double_q_learning import DoubleQ_Learning
from algorithms.general_policy_iterator_algorithms.monte_carlo import MonteCarlo
from algorithms.general_policy_iterator_algorithms.n_step_sarsa import n_step_SARSA
from algorithms.general_policy_iterator_algorithms.q_learning import Q_LearningGPI
from algorithms.general_policy_iterator_algorithms.sarsa import SARSA_GPI
from algorithms.random import RandomAlgorithm
from algorithms.r2 import Random2
from algorithms.Q_learning import Q_learning


algo_name_to_AlgoClass: Dict[str, Type[BaseRLAlgorithm]] = {
    "Random": RandomAlgorithm,
    "Random2": Random2,
    "Q_learning": Q_learning,
    "Double Q-Learning": DoubleQ_Learning,
    "SARSA": SARSA,
    "Monte Carlo": MonteCarlo,
    "SARSA GPI": SARSA_GPI,
    "n-step SARSA": n_step_SARSA,
    "Q_Learning GPI": Q_LearningGPI,
}
