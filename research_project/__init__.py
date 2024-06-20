"""project_tag"""

""" 
Other global variables
"""
from typing import List, Optional

import copy
import dataclasses
import os
import shutil
import time
import warnings
from argparse import Namespace
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from dotenv import load_dotenv
from eztils import abspath, datestr, setup_path
from eztils.argparser import HfArgumentParser, update_dataclass_defaults
from eztils.torch import seed_everything, set_gpu_mode
from meltingpot import substrate
from rich import print
from torch.utils.tensorboard import SummaryWriter

from research_project.buffer import AgentBuffer, PrincipalBuffer
from research_project.logger import Logger
from research_project.utils import *
from sen import LOG_DIR, huggingface_upload, version
from sen.neural.agent_architectures import Agent, PrincipalAgent
from sen.principal import Principal
from sen.principal.utils import vote

load_dotenv()


def get_version() -> str:
    try:
        return importlib_metadata.version("research_project")
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
__version__ = version

REPO_DIR = setup_path(Path(abspath()) / "..")
DATA_ROOT = setup_path(os.getenv("DATA_ROOT") or REPO_DIR)
RUN_DIR = LOG_DIR = Path()


def setup_experiment() -> Config:
    """
    Sets up the experiment by creating a run directory and a log directory, and creating a symlink from the repo directory to the run directory.
    """
    print("Setting up experiment...")
    global RUN_DIR
    global LOG_DIR
    # create run dir
    RUN_DIR = setup_path(DATA_ROOT / "runs")
    LOG_DIR = setup_path(RUN_DIR / datestr())

    print(f"LOG DIR: {LOG_DIR}")

    # symlink repo dir / runs to run_dir
    if not (REPO_DIR / "runs").exists() and (REPO_DIR / "runs") != RUN_DIR:
        print(f'Creating symlink from {REPO_DIR / "runs"} to {RUN_DIR}')
        (REPO_DIR / "runs").symlink_to(RUN_DIR)

    os.chdir(LOG_DIR)

    """SETUP CONFIG"""
    parser = HfArgumentParser(Config)
    parser.add_argument("-c", "--config", type=str)

    conf: Config
    extras: Namespace
    conf, extras = parser.parse_args_into_dataclasses()

    if extras.config is not None:  # parse config file
        (original_conf,) = parser.parse_json_file(extras.config)
        # reinit the parser so that the command line args overwrite the file-specified args
        parser = HfArgumentParser(update_dataclass_defaults(Config, original_conf))
        parser.add_argument("-c", "--config", type=str)
        conf, extras = parser.parse_args_into_dataclasses()

    parser.to_json([conf], LOG_DIR / "config.json")
    return conf


def main():
    args: Config = setup_experiment()
    logger = Logger()
    run_name = f"apple_picking__{args.exp_name}__{args.seed}__{int(time.time())}"

    print(f"[bold green]Welcome to research_project v{version}[/]")

    # from eztils.torch import seed_everything # install torch first to uncomment this line (by getting `poetry add eztils[torch]`` as a dependency)
    # seed_everything(conf.seed)

    if args.track:
        import wandb as wb

        wb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config=dataclasses.asdict(args),
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    seed_everything(args.seed, args.torch_deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if torch.cuda.is_available() and args.cuda:
        set_gpu_mode(True)
    print("device:", device)

    env_name = "commons_harvest__open"
    env_config = substrate.get_config(env_name)
    num_agents, num_envs, envs, principal = create_envs(args, env_config)
    voting_values, selfishness, trust = set_agent_preferences(num_agents)

    agent = Agent(envs)
    principal_agent = PrincipalAgent(num_agents)

    agent_buffer = AgentBuffer(
        num_envs,
        base_shape=(args.sampling_horizon, num_envs),
        obs_shape=envs.single_observation_space.shape,
        action_shape=envs.single_action_space.shape,
    )
    principal_buffer = PrincipalBuffer(
        num_envs,
        base_shape=(args.sampling_horizon, args.num_parallel_games),
        obs_shape=(144, 192, 3),
        action_shape=3,
        cumulative_shape=num_agents,
    )
    ctx = Context(
        args, num_agents, num_envs, envs, device, principal, agent, principal_agent
    )
    ### TODO: Ask Eddie about this (found in optimized.py)
    # if args.load_pretrained:
    #     warnings.warn("loading pretrained agents")
    #     agent.load_state_dict(torch.load("./model9399.pth"))

    ##################
    # Main Loop
    ##################

    for update in range(1, ctx.num_policy_updates_total + 1):
        #######################
        # Anneal LR and Tax Cap
        #######################
        """
        Uses:
        ctx.num_policy_updates_total
        ctx.current_episode
        ctx.tax_frac
        Sets:
        ctx.optimizer.param_groups[0]['lr']
        ctx.tax_frac
        """
        ctx.optimizer.param_groups[0]["lr"] = anneal_lr(
            update, ctx.num_policy_updates_total, args
        )
        ctx.tax_frac = anneal_tax_cap(args, ctx.current_episode, ctx.tax_frac)

        ################################
        # Collect Data For Policy Update
        ################################
        """
        USES:
        ctx.episode_step
        ctx.next_obs
        ctx.next_done
        ctx.principal_next_obs
        ctx.principal_next_done
        ctx.agent
        ctx.principal_agent
        ctx.tax_values
        ctx.principal
        ctx.num_envs
        ctx.num_agents
        ctx.device
        ctx.selfishness
        ctx.trust
        ctx.principal_tensordict["cumulative_rewards"]
        ctx.principal.objective
        ctx.tax_frac
        ctx.episode_step
        ctx.principal_episode_rewards
        ctx.episode_rewards
        ctx.episode_world_obs
        ctx.num_updates_for_this_ep

        SETS:
        ctx.agent_tensordict
        ctx.principal_tensordict
        ctx.tax_values
        ctx.next_obs
        ctx.next_done
        ctx.principal_next_obs
        ctx.principal_next_done
        ctx.principal_tensordict["cumulative_rewards"]
        ctx.episode_step
        ctx.principal_episode_rewards
        ctx.episode_rewards
        ctx.episode_world_obs
        """
        start_step, end_step = collect_data_for_policy_update(args, envs, ctx)

        ####################
        # Save Parameters
        ####################
        """
        USES:
        ctx.current_episode
        ctx.num_updates_for_this_ep
        ctx.agent_tensordict
        SET:
        None
        """
        if args.save_model and ctx.current_episode % args.save_model_freq == 0:
            save_params(ctx, run_name)

        ####################
        # Optimize Policy
        ####################
        """
        USES:
        ctx.agent
        ctx.next_obs
        ctx.next_done
        ctx.device
        ctx.agent_tensordict
        ctx.principal_agent
        ctx.principal_next_obs
        ctx.next_cumulative_reward
        ctx.principal_next_done
        ctx.principal_tensordict
        ctx.principal_advantages
        ctx.b_returns
        ctx.b_values
        ctx.optimizer
        ctx.num_updates_for_this_ep
        SETS:
        ctx.principal_advantages
        ctx.b_returns
        ctx.b_values
        """
        optimize_policy(args, ctx, envs, start_step, logger)
        # bootstrap value if not done

        ####################
        # Optimize Principal
        ####################
        """
        USES:
        ctx.principal_tensordict["obs"]
        ctx.principal_tensordict["logprobs"]
        ctx.principal_tensordict["cumulative_rewards"]
        ctx.num_agents
        ctx.principal_tensordict["actions"]
        ctx.principal_advantages
        ctx.principal_returns
        ctx.principal_tensordict["values"]
        ctx.principal_agent
        ctx.principal_optimizer
        ctx.b_values
        ctx.b_returns
        ctx.num_updates_for_this_ep
        ctx.current_episode
        SETS:
        ctx.principal_advantages
        ctx.b_returns
        ctx.b_values
        ctx.num_updates_for_this_ep
        """
        optimize_principal(args, ctx, start_step, end_step, logger)

        if ctx.num_updates_for_this_ep == ctx.num_policy_updates_per_ep:
            # episode finished

            #######
            # Log
            #######
            """
            USES:
            ctx.current_episode
            ctx.episode_world_obs
            ctx.optimizer
            ctx.episode_rewards
            ctx.num_policy_updates_per_ep
            ctx.principal_episode_rewards
            ctx.num_agents
            ctx.tax_values
            ctx.agent
            ctx.principal_agent
            SETS:
            None
            """
            logger.log(run_name, args, ctx, writer, ctx.tax_frac)

            ########################
            # Vote On New Objective
            ########################
            principal_objective = vote(voting_values)

            ################
            # Update Context
            ################
            ctx.new_episode(envs, principal_objective, args.num_parallel_games)

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
