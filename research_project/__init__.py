"""project_tag"""

""" 
Other global variables
"""
from typing import List, Optional

import dataclasses
import os
import time
from argparse import Namespace
from importlib import metadata as importlib_metadata
from pathlib import Path

import torch
from dotenv import load_dotenv
from eztils import abspath, datestr, setup_path
from eztils.argparser import HfArgumentParser, update_dataclass_defaults
from eztils.torch import seed_everything, set_gpu_mode
from rich import print
from torch.utils.tensorboard import SummaryWriter

from research_project.collection import collect_data_for_policy_update
from research_project.new_logger import MLLogger
from research_project.optimize import optimize_policy
from research_project.principal.utils import vote
from research_project.utils import *

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
        config_path = Path(extras.config)
        if not config_path.is_file():
            print(f"config file {config_path} not found. CWD: {os.getcwd()}")
        (original_conf,) = parser.parse_json_file(extras.config)
        conf = update_dataclass_defaults(Config, original_conf)
        # reinit the parser so that the command line args overwrite the file-specified args
        parser = HfArgumentParser(update_dataclass_defaults(Config, original_conf))
        parser.add_argument("-c", "--config", type=str)
        conf, extras = parser.parse_args_into_dataclasses()

    parser.to_json([conf], LOG_DIR / "config.json")

    return conf


def main():
    args: Config = setup_experiment()
    logger = MLLogger(cfg=args)
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

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if torch.cuda.is_available() and args.cuda:
        set_gpu_mode(True)
    print("device:", device)

    ctx, envs = set_context(args, device)
    voting_values, trust = set_agent_preferences(ctx.num_agents)

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

        ctx.optimizer.param_groups[0]["lr"] = anneal_lr(
            update, ctx.num_policy_updates_total, args
        )
        ctx.tax_frac = anneal_tax_cap(args, ctx.current_episode, ctx.tax_frac)

        ################################
        # Collect Data For Policy Update
        ################################

        start_step, end_step = collect_data_for_policy_update(
            args=args, ctx=ctx, envs=envs, logger=logger
        )

        ####################
        # Save Parameters
        ####################

        if args.save_model and ctx.current_episode % args.save_model_freq == 0:
            save_params(ctx=ctx, run_name=run_name)

        ####################
        # Optimize Policy
        ####################

        optimize_policy(args, ctx, ctx.agent, logger, ctx.agent_buffer)
        if not args.LLM:
            optimize_policy(
                args, ctx, ctx.principal_agent, logger, ctx.principal_buffer
            )

        if ctx.num_updates_for_this_ep == ctx.num_policy_updates_per_ep:
            # episode finished

            #######
            # Log
            #######
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
