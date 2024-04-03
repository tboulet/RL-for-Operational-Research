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


def instantiate_class(**kwargs) -> Any:
    """Instantiate a class from a dictionnary that contains a key "class_string" with the format "path.to.module:ClassName"
    and that contains other keys that will be passed as arguments to the class constructor

    Args:
        config (dict): the configuration dictionnary
        **kwargs: additional arguments to pass to the class constructor

    Returns:
        Any: the instantiated class
    """
    assert (
        "class_string" in kwargs
    ), "The class_string should be specified in the config"
    class_string: str = kwargs["class_string"]
    module_name, class_name = class_string.split(":")
    module = importlib.import_module(module_name)
    Class = getattr(module, class_name)
    object_config = kwargs.copy()
    object_config.pop("class_string")
    return Class(**object_config)


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


def get_softmax_probs(
    logits: Union[np.ndarray, Dict[str, float]],
    temperature: float = 1,
) -> np.ndarray:
    """Compute the softmax probabilities from the logits

    Args:
        logits (Union[np.ndarray, Dict[str, float]]): the logits, as a numpy array of shape (n_actions,) or (n_states, n_actions), or as a dictionnary dict[state][action] = logit or dict[action] = logit
        temperature (float): the temperature of the softmax, if any

    Returns:
        np.ndarray: the softmax probabilities
    """
    if isinstance(logits, np.ndarray):
        # Case 1 : logits is a numpy array of shape (n_actions,) or (n_states, n_actions)
        logits = logits / temperature
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs /= np.sum(probs, axis=-1, keepdims=True)
    elif isinstance(logits, dict):
        if isinstance(list(logits.values())[0], dict):
            # Case 2 : logits is a dictionnary dict[state][action] = logits
            probs = {
                state: get_softmax_probs(logit, temperature=temperature)
                for state, logit in logits.items()
            }
        elif isinstance(list(logits.values())[0], (int, float)):
            # Case 3 : logits is a dictionnary dict[action] = logit
            probs = {
                action: np.exp(logit / temperature) for action, logit in logits.items()
            }
            total = sum(probs.values())
            probs = {action: prob / total for action, prob in probs.items()}
    else:
        raise ValueError("The logits should be either a numpy array or a dictionnary")
    return probs
