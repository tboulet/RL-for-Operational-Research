from typing import Dict, Type, Any, Tuple
from environments.base_environment import BaseOREnvironment
from environments.toy import ToyExampleEnvironment
import gym

env_name_to_EnvClass: Dict[str, Type[BaseOREnvironment]] = {
    "Toy Example": ToyExampleEnvironment,
}
