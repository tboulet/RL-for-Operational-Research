# Logging
import os
import sys
import wandb
from tensorboardX import SummaryWriter
import ast
# Config system
import hydra
from omegaconf import OmegaConf, DictConfig

# Utils
from tqdm import tqdm
import datetime
import time
from typing import Dict, Type, Any, Tuple
import cProfile

# ML libraries
import random
import numpy as np
from environments.base_environment import BaseOREnvironment

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

# Project imports
from environments.wrappers.reward_normalizer import get_normalized_reward_env_class
from environments.wrappers.sparsifier import get_sparsified_env_class
from src.time_measure import RuntimeMeter
from src.utils import (
    get_normalized_performance,
    try_get,
    try_get_seed,
)
from environments import env_name_to_EnvClass
from algorithms import algo_name_to_AlgoClass
import subprocess

def main():
    pass

if __name__ == "__main__":
    for i in range(9):
        subprocess.run(["python", "run.py"])