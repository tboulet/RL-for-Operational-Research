from enum import Enum
from typing import List, Tuple, Dict, Type, Union, Any


class State:
    pass


class Action:
    pass

StateValues = Dict[State, float]
QValues = Dict[State, Dict[Action, float]]