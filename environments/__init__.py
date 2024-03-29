from typing import Dict, Type, Any, Tuple
from environments.base_environment import BaseOREnvironment
from environments.flappy_bird import FlappyBirdEnv
from environments.toy import ToyExampleEnvironment
from environments.facility_location_problem import FacilityLocationProblemEnvironment
from environments.knapsack import knapsack
from environments.bin_packing import BinPacking
import gym

env_name_to_EnvClass: Dict[str, Type[BaseOREnvironment]] = {
    "Toy Example": ToyExampleEnvironment,
    "Facility Location Problem": FacilityLocationProblemEnvironment,
    "knapsack": knapsack,
    "Flappy Bird" : FlappyBirdEnv,
    "bin_packing": BinPacking,
}
