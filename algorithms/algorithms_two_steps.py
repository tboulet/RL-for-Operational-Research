"""Util class for algorithm that learns from a two-step transition, i.e., (state, action, reward, next_state, done, next_action, next_reward, next_next_state, next_done).

Such algorithms include :
    - SARSA
    - 2-step Q-learning
    - 2-step Expected SARSA
    - 2-step Sampled Expected SARSA
"""

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



class AlgorithmTwoSteps:
    raise NotImplementedError