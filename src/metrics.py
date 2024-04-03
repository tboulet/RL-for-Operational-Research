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


class DictAverager:
    """A class to average metrics and store them in a dictionary."""

    def create_dict_statistical(
        self,
        value: Union[float, int],
        keys_statistical: List[str] = ["mean", "count"],
    ) -> Dict[str, Union[float, int]]:
        """Initializes a dictionary to store the metrics.

        Args:
            value (Union[float, int]): the value to initialize the dictionary with

        Returns:
            Dict[str, Union[float, int]]: a dictionnary containing statistical information about the metrics (mean, std, min, max, count)
        """
        dict_statistical: Dict[str, Union[float, int]] = {}
        assert not "mean" in keys_statistical or "count" in keys_statistical, "If you want to compute the mean, you need to compute the count"
        assert not "std" in keys_statistical or "mean" in keys_statistical, "If you want to compute the std, you need to compute the mean"
        assert "std" not in keys_statistical, "The std is not implemented yet" # TODO: implement the std
        for key in keys_statistical:
            if key in ["mean", "min", "max"]:
                dict_statistical[key] = value
            elif key in ["std"]:
                dict_statistical[key] = 0
            elif key in ["count"]:
                dict_statistical[key] = 1
            else:
                raise ValueError(f"key {key} not recognized")
        return dict_statistical

    def __init__(self) -> None:
        self.metrics: Dict[str, Dict[str, Union[float, int]]] = {}

    def add(
        self,
        key: str,
        value: Union[float, int],
        keys_statistical: List[str] = ["mean", "count"],
    ) -> None:
        """Add a value to the metrics.

        Args:
            key (str): the key of the metric
            value (Union[float, int]): the value of the metric
        """
        if key not in self.metrics:
            self.metrics[key] = self.create_dict_statistical(
                value=value, keys_statistical=keys_statistical
            )
        else:
            for key_stat in keys_statistical:
                if key_stat == "mean":
                    n = self.metrics[key]["count"]
                    mean = self.metrics[key]["mean"]
                    self.metrics[key]["mean"] = (n * mean + value) / (n + 1)
                elif key_stat == "std":
                    raise NotImplementedError("The std is not implemented yet")
                elif key_stat == "min":
                    self.metrics[key]["min"] = min(self.metrics[key]["min"], value)
                elif key_stat == "max":
                    self.metrics[key]["max"] = max(self.metrics[key]["max"], value)
                elif key_stat == "count":
                    self.metrics[key]["count"] += 1
                else:
                    raise ValueError(f"key_stat {key_stat} not recognized")
    
    def add_dict(self, dict_metrics : Dict[str, Union[float, int]], keys_statistical: List[str] = ["mean", "count"]) -> None:
        for key, value in dict_metrics.items():
            self.add(key, value, keys_statistical)
    
    def get(self, metric_key : str, key_stat : str = "mean") -> Union[float, int]:
        return self.metrics[metric_key][key_stat]
    
    def get_mean(self, metric_key : str) -> Union[float, int]:
        return self.get(metric_key, "mean")
                    
    def get_dict(self, key_stat : str = "mean") -> Dict[str, Union[float, int]]:
        return {key: self.get(key, key_stat) for key in self.metrics.keys()}