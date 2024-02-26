from typing import Dict, Type, Any, Tuple
from algorithms.base_algorithm import BaseRLAlgorithm
from algorithms.random import RandomAlgorithm
from algorithms.r2 import Random2


algo_name_to_AlgoClass: Dict[str, Type[BaseRLAlgorithm]] = {
    "Random": RandomAlgorithm,
    "Random2": Random2,
}
