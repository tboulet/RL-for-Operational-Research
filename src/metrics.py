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
from src.constants import INF

from src.schedulers import Scheduler
from src.typing import Action, QValues, State


def get_scheduler_metrics_of_object(obj: Any) -> Dict[str, Any]:
    """Get the numerical values of all scheduler objects that are attributes of that object.

    Args:
        obj (Any): the object to inspect

    Returns:
        Dict[str, Any]: the metrics to log
    """
    metrics = {}
    for attr in dir(obj):
        attribute = getattr(obj, attr)
        if isinstance(attribute, Scheduler):
            metrics[attr] = attribute.get_value()
            metrics[attr + "_step"] = attribute.get_step()
    return metrics


def get_q_values_metrics(
    q_values: Dict[State, Dict[Action, float]],
    n_max_states_to_log: int = INF,
) -> Dict[str, Any]:
    """Get the numerical values of the Q values as a dictionary of metrics.
    For limiting the logging, the maximum number of Q values to log per state and the maximum number of Q values to log can be set.
    In that case, the choice of which Q values to log is arbitrary.

    Args:
        q_values (Dict[State, Dict[Action, float]]): the Q values to log
        n_max_states_to_log (int, optional): the maximum number of states' Q values to log. Defaults to INF.

    Returns:
        Dict[str, Any]: the metrics to log
    """
    if len(q_values) > n_max_states_to_log:
        states = list(q_values.keys())
        states = random.choices(states, k=n_max_states_to_log)
        return {
            f"q_values/Q(s={state}, a={action})": value
            for state in states
            for action, value in q_values[state].items()
        }
    else:
        return {
            f"q_values/Q(s={state}, a={action})": value
            for state, actions in q_values.items()
            for action, value in actions.items()
        }
