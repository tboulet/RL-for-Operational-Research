from typing import Dict, Type, Any, Tuple

from algorithms.base.base_algorithm import BaseRLAlgorithm

from algorithms.random import RandomAlgorithm
from algorithms.q_learning import Q_Learning
from algorithms.double_q_learning import (
    DoubleQ_Learning,
)
from algorithms.sarsa import SARSA
from algorithms.n_step_sarsa import n_step_SARSA
from algorithms.sarsa_lambda import SARSA_Lambda
from algorithms.monte_carlo import MonteCarlo
from algorithms.reinforce import REINFORCE
from algorithms.deep_q_learning import DeepQ_Learning


algo_name_to_AlgoClass: Dict[str, Type[BaseRLAlgorithm]] = {
    "Random": RandomAlgorithm,
    "Q-Learning": Q_Learning,
    "Double Q-Learning": DoubleQ_Learning,
    "SARSA": SARSA,
    "n-step SARSA": n_step_SARSA,
    "SARSA Lambda": SARSA_Lambda,
    "Monte Carlo": MonteCarlo,
    "REINFORCE": REINFORCE,
    "Deep-Q-Learning": DeepQ_Learning
}
