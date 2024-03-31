from typing import Dict, Type, Any, Tuple
from environments.base_environment import BaseOREnvironment
from environments.flappy_bird import FlappyBirdEnv
from environments.toy import ToyExampleEnvironment
from environments.facility_location_problem import FacilityLocationProblemEnvironment
from environments.knapsack import KnapsackEnv
from environments.bin_packing import BinPacking
from environments.knapsack_deep import KnapsackEnvDeep
from environments.bin_packing_deep import BinPackingDeep
from environments.facility_location_problem_deep import FacilityLocationProblemEnvironmentDeep
import gym

env_name_to_EnvClass: Dict[str, Type[BaseOREnvironment]] = {
    "Toy Example": ToyExampleEnvironment,
    "Facility Location Problem": FacilityLocationProblemEnvironment,
    "Knapsack": KnapsackEnv,
    "Flappy Bird": FlappyBirdEnv,
    "Bin Packing": BinPacking,
    "Knapsack Deep": KnapsackEnvDeep,
    "Bin Packing Deep": BinPackingDeep,
    "Facility Location Problem Deep": FacilityLocationProblemEnvironmentDeep,

}
