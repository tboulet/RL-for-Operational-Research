from typing import Dict, Type, Any, Tuple
from algorithms.base_algorithm import BaseRLAlgorithm
from algorithms.random import RandomAlgorithm
from algorithms.r2 import Random2
from algorithms.Q_learning import Q_learning


algo_name_to_AlgoClass: Dict[str, Type[BaseRLAlgorithm]] = {
    "Random": RandomAlgorithm,
    "Random2": Random2,
    "Q_learning": Q_learning,
}
