# Logging
import os
import wandb
from tensorboardX import SummaryWriter

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

# Project imports
from src.time_measure import RuntimeMeter
from src.utils import try_get_seed
from environments import env_name_to_EnvClass
from algorithms import algo_name_to_AlgoClass


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


@hydra.main(config_path="configs", config_name="config_default.yaml")
def main(config: DictConfig):
    """The main function of the project. It initializes the environment and the algorithm, then runs the training loop.

    Args:
        config (DictConfig): the configuration of the project. This will be imported from the config_default.yaml file in the configs folder,
        thanks to the @hydra.main decorator.
    """
    print("Configuration used :")
    print(OmegaConf.to_yaml(config))

    # Get the config values from the config object.
    algo_name: str = config["algo"]["name"]
    env_name: str = config["env"]["name"]
    n_episodes_training: int = config["n_episodes_training"]
    do_cli: bool = config["do_cli"]
    cli_frequency_episode = config["cli_frequency_episode"]
    do_wandb: bool = config["do_wandb"]
    do_tb: bool = config["do_tb"]
    do_tqdm: bool = config["do_tqdm"]
    render_config: dict = config["render_config"]

    # Set the seeds
    seed = try_get_seed(config)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Using seed: {seed}")

    # Create the environment
    print("Initializing the environment...")
    EnvClass = env_name_to_EnvClass[env_name]
    env = EnvClass(config["env"]["config"])

    # Create the algorithm
    print("Initializing the algorithm...")
    AlgoClass = algo_name_to_AlgoClass[algo_name]
    algo = AlgoClass(config=config["algo"]["config"])

    # Initialize loggers
    run_name = f"[{algo_name}]_[{env_name}]_{datetime.datetime.now().strftime('%dth%mmo_%Hh%Mmin%Ss')}_seed{seed}"
    os.makedirs("logs", exist_ok=True)
    print(f"\nStarting run {run_name}")
    if do_wandb:
        run = wandb.init(
            name=run_name,
            config=config,
            **config["wandb_config"],
        )
    if do_tb:
        tb_writer = SummaryWriter(log_dir=f"tensorboard/{run_name}")

    # Training loop
    for episode in tqdm(range(n_episodes_training), disable=not do_tqdm):

        # Reset the environment
        with RuntimeMeter("env reset") as rm:
            state, info = env.reset(seed=seed)
            available_actions = env.get_available_actions(state=state)
            done = False
            episodic_reward = 0
            step = 0

        # Render the environment
        with RuntimeMeter("env render") as rm:
            try_render(env, episode, step, render_config, done)

        # Play one episode
        while not done:
            with RuntimeMeter("agent act") as rm:
                action = algo.act(state=state, available_actions=available_actions)

            with RuntimeMeter("env step") as rm:
                next_state, reward, is_trunc, done, info = env.step(action)
                next_available_actions = env.get_available_actions(state=next_state)
                episodic_reward += reward

            with RuntimeMeter("env render") as rm:
                try_render(env, episode, step, render_config, done)
                

            with RuntimeMeter("agent update") as rm:
                algo.update(state, action, reward, next_state, done)

            state = next_state
            available_actions = next_available_actions
            step += 1

        # Close the environment
        with RuntimeMeter("env close") as rm:
            env.close()

        # Log metrics.
        metrics = {}
        runtime_agent_total_in_ms = int(
            (rm.get_stage_runtime("agent act") + rm.get_stage_runtime("agent update"))
            * 1000
        )
        metrics.update(
            {
                f"runtime {stage_name}": value
                for stage_name, value in rm.get_stage_runtimes().items()
            }
        )
        metrics["runtime total"] = rm.get_total_runtime()
        metrics["runtime agent total in ms"] = runtime_agent_total_in_ms
        metrics["episode"] = episode
        metrics["episodic reward"] = episodic_reward

        with RuntimeMeter("log") as rm:
            # Log on WandB
            if do_wandb:
                wandb.log(metrics, step=episode)
            # Log on Tensorboard
            for metric_name, metric_value in metrics.items():
                if do_tb:
                    tb_writer.add_scalar(
                        tag=f"metrics/{metric_name}",
                        scalar_value=metric_value,
                        global_step=episode,
                    )
            # Log on CLI
            if do_cli and (episode % cli_frequency_episode == 0):
                print(f"Metric results at episode {episode}: {metrics}")

    # Finish the WandB run.
    if do_wandb:
        run.finish()


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()
    pr.dump_stats("logs/profile_stats.prof")
    print("\nProfile stats dumped to profile_stats.prof")
    print(
        "You can visualize the profile stats using snakeviz by running 'snakeviz logs/profile_stats.prof'"
    )
