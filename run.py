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

OmegaConf.register_new_resolver("eval", eval)


def try_render(
    env: BaseOREnvironment,
    episode: int,
    step: int,
    render_config: dict,
    done: bool,
) -> None:
    """Try to render the environment, depending on the render_config and the current episode and step.

    Args:
        env (BaseOREnvironment): the env to render at that step
        episode (int): the current episode
        step (int): the current step among the episode
        render_config (dict): the render configuration
        done (bool): whether the episode is done or not
    """
    # Check if we should render
    if not render_config["do_render"]:
        return
    if not episode % render_config["frequency_episode"] == 0:
        return
    if (not done) and (not step % render_config["frequency_step"] == 0):
        return
    # Render the environment and wait a bit
    env.render()
    time.sleep(render_config["delay"])


@hydra.main(config_path="configs", config_name="default.yaml")
def main(config_omega: DictConfig):
    """The main function of the project. It initializes the environment and the algorithm, then runs the training loop.

    Args:
        config_omega (DictConfig): the configuration of the project. This will be imported from the config_default.yaml file in the configs folder,
        thanks to the @hydra.main decorator.
    """
    print("Configuration used :")
    print(OmegaConf.to_yaml(config_omega))
    config = OmegaConf.to_container(config_omega, resolve=True)
    
    # Extract the name of the environment and the algorithm
    algo_name: str = config["algo"]["name"]
    algo_name_full = try_get(config["algo"], "name_full", default=algo_name)
    config["algo"]["name_full"] = algo_name_full
    env_name : str = config["env"]["name"]
    env_name_full = try_get(config["env"], "name_full", default=env_name)
    config["env"]["name_full"] = env_name_full
    # Hyperparameters of the RL loop
    n_max_episodes_training: int = try_get(
        config, "n_max_episodes_training", sys.maxsize
    )
    n_max_steps_training: int = try_get(config, "n_max_steps_training", sys.maxsize)
    eval_frequency_episode: int = try_get(config, "eval_frequency_episode", None)
    render_config_train: dict = config["render_config_train"]
    render_config_eval: dict = config["render_config_eval"]
    # Logging
    do_wandb: bool = config["do_wandb"]
    wandb_config: dict = config["wandb_config"]
    do_tb: bool = config["do_tb"]
    do_cli: bool = config["do_cli"]
    cli_frequency_episode = config["cli_frequency_episode"]
    do_tqdm: bool = config["do_tqdm"]

    # Set the seeds
    seed = try_get_seed(config)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Using seed: {seed}")

    # Create the environment
    print("Initializing the environment...")
    EnvClass = env_name_to_EnvClass[env_name]
    if config["do_sparsify_reward"]:
        EnvClass = get_sparsified_env_class(EnvClass)
    if config["do_normalize_reward"]:    
        EnvClass = get_normalized_reward_env_class(EnvClass)
    env = EnvClass(config["env"]["config"])
        
    # Create the algorithm
    print("Initializing the algorithm...")
    AlgoClass = algo_name_to_AlgoClass[algo_name]
    algo = AlgoClass(config=config["algo"]["config"])


    

    # Initialize loggers
    run_name = f"[{algo_name_full}]_[{env_name_full}]_{datetime.datetime.now().strftime('%dth%mmo_%Hh%Mmin%Ss')}_seed{seed}"
    os.makedirs("logs", exist_ok=True)
    print(f"\nStarting run {run_name}")
    if do_wandb:
        run = wandb.init(
            name=run_name,
            config=config,
            **wandb_config,
        )
    if do_tb:
        tb_writer = SummaryWriter(log_dir=f"tensorboard/{run_name}")

    # Reinforcement Learning loop
    episode_train = 0
    total_steps_train = 0
    episode_eval = 0
    total_steps_eval = 0
    metrics_from_algo: Dict[str, float] = None
    training_episode_bar = tqdm(
        total=n_max_episodes_training,
        disable=not do_tqdm and n_max_episodes_training != sys.maxsize,
    )
    while (
        episode_train < n_max_episodes_training
        and total_steps_train < n_max_steps_training
    ):
        # Set the settings whether we are in eval mode or not
        if eval_frequency_episode is not None and (episode_train + episode_eval) % eval_frequency_episode == 0:
            is_eval = True
            mode = "eval"
            render_config = render_config_eval
        else:
            is_eval = False
            mode = "train"
            render_config = render_config_train

        # Reset the environment
        with RuntimeMeter(f"{mode}/env reset") as rm:
            state, info = env.reset(seed=seed)
            available_actions = env.get_available_actions(state=state)
            done = False
            episodic_reward = 0
            step = 0

        # Render the environment
        with RuntimeMeter(f"{mode}/env render") as rm:
            try_render(
                env=env,
                episode=episode_eval if is_eval else episode_train,
                step=step,
                render_config=render_config,
                done=done,
            )

        # Play one episode
        while not done and total_steps_train < n_max_steps_training:
            with RuntimeMeter(f"{mode}/agent act") as rm:

                
                action = algo.act(
                    state=state,
                    available_actions=available_actions,
                    is_eval=is_eval,
                )


            with RuntimeMeter(f"{mode}/env step") as rm:
                next_state, reward, is_trunc, done, info = env.step(action)
                next_available_actions = env.get_available_actions(state=next_state)
                #print(next_available_actions)
                episodic_reward += reward

            with RuntimeMeter(f"{mode}/env render") as rm:
                try_render(
                    env=env,
                    episode=episode_eval if is_eval else episode_train,
                    step=step,
                    render_config=render_config,
                    done=done,
                )

            if not is_eval:
                with RuntimeMeter(f"{mode}/agent update") as rm:

                    metrics_from_algo = algo.update(
                        state, action, reward, next_state, done
                    )


            # Update the variables
            state = next_state
            available_actions = next_available_actions
            step += 1
            if is_eval:
                total_steps_eval += 1
            else:
                total_steps_train += 1

        # Close the environment
        with RuntimeMeter(f"{mode}/env close") as rm:
            env.close()

        # Compute the metrics and log them
        with RuntimeMeter(f"{mode}/log") as rm:
            metrics = {}
            runtime_training_agent_total_in_ms = int(
                (
                    rm.get_stage_runtime("train/agent act")
                    + rm.get_stage_runtime("train/agent update")
                )
                * 1000
            )
            # Add the episodic reward
            metrics[f"{mode}/episodic reward"] = episodic_reward
            # If the env implement a notion of optimal and worst reward, add the normalized performance
            optimal_reward = env.get_optimal_reward()
            worst_reward = env.get_worst_reward()
            normalized_performance = get_normalized_performance(
                episodic_reward=episodic_reward,
                optimal_reward=optimal_reward,
                worst_reward=worst_reward,
            )
            if normalized_performance is not None:
                metrics[f"{mode}/normalized performance"] = normalized_performance
            # Add the runtime of the different stages
            metrics.update(
                {
                    f"runtime/{stage_name}": value
                    for stage_name, value in rm.get_stage_runtimes().items()
                }
            )
            # Add various information
            metrics["other/runtime total"] = rm.get_total_runtime()
            metrics["other/runtime training agent total in ms"] = (
                runtime_training_agent_total_in_ms
            )
            metrics["other/episode training"] = episode_train
            metrics["other/step training"] = total_steps_train
            # Add agent-specific metrics
            if metrics_from_algo is not None:
                metrics.update({f"agent/{k}": v for k, v in metrics_from_algo.items()})

            # Log on WandB
            if do_wandb:
                wandb.log(metrics, step=episode_train)
            # Log on Tensorboard
            for metric_name, metric_value in metrics.items():
                if do_tb:
                    tb_writer.add_scalar(
                        tag=metric_name,
                        scalar_value=metric_value,
                        global_step=episode_train,
                    )
            # Log on CLI
            if do_cli and (episode_train % cli_frequency_episode == 0):
                print(f"Metric results at episode {episode_train}: {metrics}")

        # Update the variables
        if is_eval:
            episode_eval += 1
        else:
            episode_train += 1
            training_episode_bar.update(1)

    # Finish the WandB run.
    if do_wandb:
        run.finish()
    
    if 'show_moche' in config["algo"]["config"]:
        if config["algo"]["config"]["show_moche"] is True:
            algo.show()


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()
    pr.dump_stats("logs/profile_stats.prof")
    print("\nProfile stats dumped to profile_stats.prof")
    print(
        "You can visualize the profile stats using snakeviz by running 'snakeviz logs/profile_stats.prof'"
    )
