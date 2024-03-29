import importlib
from re import T
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


def to_numeric(x: Union[int, float, str, None]) -> Union[int, float]:
    if isinstance(x, int) or isinstance(x, float):
        return x
    elif isinstance(x, str):
        return float(x)
    elif x == "inf":
        return float("inf")
    elif x == "-inf":
        return float("-inf")
    elif x is None:
        return None
    else:
        raise ValueError(f"Cannot convert {x} to numeric")


def try_get_seed(config: Dict) -> int:
    """Will try to extract the seed from the config, or return a random one if not found

    Args:
        config (Dict): the run config

    Returns:
        int: the seed
    """
    try:
        seed = config["seed"]
        if not isinstance(seed, int):
            seed = np.random.randint(0, 1000)
    except KeyError:
        seed = np.random.randint(0, 1000)
    return seed


def try_get(dictionnary: Dict, key: str, default: Union[int, float, str, None]) -> Any:
    """Will try to extract the key from the dictionary, or return the default value if not found
    or if the value is None

    Args:
        x (Dict): the dictionary
        key (str): the key to extract
        default (Union[int, float, str, None]): the default value

    Returns:
        Any: the value of the key if found, or the default value if not found
    """
    try:
        return dictionnary[key] if dictionnary[key] is not None else default
    except KeyError:
        return default


def instantiate_class(config: dict) -> Any:
    """Instantiate a class from a dictionnary that contains a key "class_string" with the format "path.to.module:ClassName"
    and that contains other keys that will be passed as arguments to the class constructor

    Args:
        config (dict): the configuration dictionnary

    Returns:
        Any: the instantiated class
    """
    assert (
        "class_string" in config
    ), "The class_string should be specified in the config"
    class_string: str = config["class_string"]
    module_name, class_name = class_string.split(":")
    module = importlib.import_module(module_name)
    Class = getattr(module, class_name)
    return Class(**{k: v for k, v in config.items() if k != "class_string"})


def get_normalized_performance(
    episodic_reward: float,
    optimal_reward: Optional[float] = None,
    worst_reward: Optional[float] = None,
):
    assert isinstance(
        episodic_reward, (int, float)
    ), "The episodic reward should be a number"
    if optimal_reward is None or worst_reward is None:
        return None
    assert isinstance(
        optimal_reward, (int, float)
    ), "The optimal reward should be a number, or None"
    assert isinstance(
        worst_reward, (int, float)
    ), "The worst reward should be a number, or None"
    assert (
        worst_reward <= optimal_reward
    ), "The worst reward should be less than the optimal reward"
    if optimal_reward == worst_reward:
        return None
    return (episodic_reward - worst_reward) / (optimal_reward - worst_reward)
