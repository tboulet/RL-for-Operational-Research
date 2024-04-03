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
from typing import Dict, List, Optional, Type, Any, Tuple
import cProfile

# ML libraries
import random
import numpy as np

# File specific
from abc import ABC, abstractmethod
from src.constants import INF
from src.schedulers import Scheduler

# Project imports
from src.typing import State, Action
from src.utils import try_get


def initialize_scalar_value(
        method_value_initialization: str, 
        typical_return : Optional[float] = 0,
        typical_return_std : Optional[float] = 1, 
    ) -> float:
    """Initialize a scalar value based on the method.

    Args:
        method_value_initialization (str): the method to initialize the value, one of ['random', 'zero', 'optimistic']
        typical_return (Optional[float], optional): the typical return value. Defaults to 0.
        typical_return_std (Optional[float], optional): the typical return standard deviation. Defaults to 1.
        
    Returns:
        float: the initialized value
    """
    if method_value_initialization == "random":
        # Random initialization : Q(s, a) ~ N(typical_return, typical_return_std)
        return np.random.normal(typical_return, typical_return_std)

    elif method_value_initialization == "zero":
        # Zero initialization
        return 0.0

    elif method_value_initialization == "optimistic":
        # Optimistic initialization
        return typical_return + 5 * typical_return_std

    else:
        raise ValueError(
            f"The method of value initialization '{method_value_initialization}' is not recognized. Please use one of the following methods: 'random', 'zero', 'optimistic'."
        )
             
        
def initialize_tabular_q_values(
        method_q_value_initialization: str, 
        typical_return : Optional[float] = 0,
        typical_return_std : Optional[float] = 1, 
    ) -> Dict[State, Dict[Action, float]]:
    """Initialize a tabular Q-values object based on the method, as an hash map of hash maps.

    Args:
        method_q_value_initialization (str): the method to initialize the Q-values, one of ['random', 'zero', 'optimistic']
        typical_return (Optional[float], optional): the typical return value. Defaults to 0.
        typical_return_std (Optional[float], optional): the typical return standard deviation. Defaults to 1.
        
    Returns:
        Dict[State, Dict[Action, float]]: the initialized Q-values
    """
    def initialize_scalar_value_func():
        return initialize_scalar_value(
            method_value_initialization=method_q_value_initialization,
            typical_return=typical_return,
            typical_return_std=typical_return_std,
        )
    return defaultdict(lambda: defaultdict(initialize_scalar_value_func))
        
        
def initialize_tabular_v_values(
        method_v_value_initialization: str, 
        typical_return : Optional[float] = 0,
        typical_return_std : Optional[float] = 1, 
    ) -> Dict[State, float]:
    """Initialize a tabular V-values object based on the method, as an hash map.

    Args:
        method_v_value_initialization (str): the method to initialize the V-values, one of ['random', 'zero', 'optimistic']
        typical_return (Optional[float], optional): the typical return value. Defaults to 0.
        typical_return_std (Optional[float], optional): the typical return standard deviation. Defaults to 1.
        
    Returns:
        Dict[State, float]: the initialized V-values
    """
    def initialize_scalar_value_func():
        return initialize_scalar_value(
            method_value_initialization=method_v_value_initialization,
            typical_return=typical_return,
            typical_return_std=typical_return_std,
        )
    return defaultdict(initialize_scalar_value_func)
    


    