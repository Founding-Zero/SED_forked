"""project_tag"""

""" 
Other global variables
"""
import os
import time
import copy
import shutil
import warnings
from argparse import Namespace
from pathlib import Path
from typing import List, Optional

import dataclasses
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from importlib import metadata as importlib_metadata
from torch.utils.tensorboard import SummaryWriter

from dotenv import load_dotenv
from eztils import abspath, datestr, setup_path
from eztils.argparser import HfArgumentParser, update_dataclass_defaults
from eztils.torch import seed_everything
from rich import print
from meltingpot import substrate
import wandb

from research_project import utils
from sen import LOG_DIR, huggingface_upload, version
from sen.neural.agent_architectures import Agent, PrincipalAgent
from sen.principal import Principal
from sen.principal.utils import vote
from utils import *

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
    print("device:", device)

    # set up envs
    env_name = "commons_harvest__open"
    env_config = substrate.get_config(env_name)
    num_players = len(env_config.default_player_roles)
    num_agents, num_envs, envs, principal = create_env(args, env_config, num_players)

    principal = Principal(num_players, args.num_parallel_games, "egalitarian")

    env = utils.parallel_env(
        max_cycles=args.sampling_horizon, env_config=env_config, principal=principal
    )

    num_agents = env.max_num_agents
    num_envs = args.num_parallel_games * num_agents
    voting_values, selfishness, trust = set_agent_preferencts(num_agents)


    
    envs = set_up_envs(env, args.num_frames, args.num_parallel_games)
    ctx = Context(args, num_agents, num_envs, envs, device)
    

    for update in range(1, ctx.num_policy_updates_total + 1):
        # annealing the rate if instructed to do so
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / ctx.num_policy_updates_total
            lrnow = frac * args.learning_rate
            ctx.optimizer.param_groups[0]["lr"] = lrnow

        # annealing tax controlling multiplier
        if args.anneal_tax:
            tax_frac = 0.1 + 0.9 * min(
                (ctx.current_episode - 1.0)
                / (args.num_episodes * args.tax_annealment_proportion),
                1,
            )

        # collect data for policy update
        start_step = episode_step
        for step in range(0, args.sampling_horizon):
            if next_obs.shape[3] != 19:
                warnings.warn(
                    "hardcoded value of 12 RGB channels - check RBG/indicator channel division here"
                )
            num_rgb_channels = 12
            # we only divide the 4 stack frames x 3 RGB channels - NOT the agent indicators
            next_obs[:, :, :, :num_rgb_channels] /= 255.0
            ctx.a_obs[step] = next_obs
            ctx.a_dones[step] = next_done
            ctx.p_obs[step] = principal_next_obs
            ctx.p_dones[step] = principal_next_done

            with torch.no_grad():
                action, logprob, _, value = ctx.agent.get_action_and_value(next_obs)
                ctx.a_values[step] = value.flatten()
            ctx.a_actions[step] = action
            ctx.a_logprobs[step] = logprob

            with torch.no_grad():
                (
                    principal_action,
                    principal_logprob,
                    _,
                    principal_value,
                ) = ctx.principal_agent.get_action_and_value(
                    principal_next_obs, next_cumulative_reward
                )
                ctx.p_values[step] = principal_value.flatten()

            if episode_step % args.tax_period == 0:
                # this `principal_action` is the one that was fed cumulative reward of last step of previous tax period
                # so it is acting on the full wealth accumulated last tax period and an observation of the last frame
                ctx.p_actions[step] = principal_action
                ctx.p_logprobs[step] = principal_logprob
                principal.update_tax_vals(principal_action)
                tax_values.append(copy.deepcopy(principal.tax_vals))
            else:
                ctx.p_actions[step] = torch.tensor([11] * 3)
                ctx.p_actions[step] = torch.full((args.num_parallel_games, 3), 11)
                ctx.p_logprobs[step] = torch.zeros(args.num_parallel_games)

            """
            NOTE: info has been changed to return a list of entries for each
                  environment (over num_agents and num_parallel_games), with
                  each entry being a tuple of the old info dict (asfaik always
                  empty until last step when it gets a 'terminal_observation'),
                  the world observation numpy array and the nearby player array.
                  IMPORTANT:
                  info is a list of environment, not agents.
                  If you are playing 2 simultaneous games of seven players, info
                  will be a list of length 14. In this, the first seven entries
                  will have the same info[i][1] world observation, and so will the
                  next seven - but the two will differ between each other.
            """
            next_obs, extrinsic_reward, done, info = envs.step(action.cpu().numpy())
            principal.report_reward(extrinsic_reward)

            # mix personal and nearby a_rewards
            intrinsic_reward = np.zeros_like(extrinsic_reward)
            nearby = torch.stack(
                [torch.Tensor(info[i][2]) for i in range(0, num_envs)]
            ).to(device)
            for game_id in range(args.num_parallel_games):
                for player_id in range(num_agents):
                    env_id = player_id + game_id * num_agents
                    w = selfishness[player_id]
                    #! TODO fix nearby_reward = sum(nearby[env_id] * game_reward)
                    nearby_reward = 0
                    intrinsic_reward[env_id] = (
                        w * extrinsic_reward[env_id] + (1 - w) * nearby_reward
                    )

            # make sure tax is applied after extrinsic reward is used for intrinsic reward calculation
            if (episode_step + 1) % args.tax_period == 0:
                # last step of tax period
                taxes = principal.end_of_tax_period()
                extrinsic_reward -= tax_frac * np.array(list(taxes.values())).flatten()

            reward = np.zeros_like(extrinsic_reward)
            for env_id in range(len(reward)):
                player_id = env_id % num_agents
                v = trust[player_id]
                reward[env_id] = (
                    v * extrinsic_reward[env_id] + (1 - v) * intrinsic_reward[env_id]
                )

            principal_next_obs = torch.stack(
                [torch.Tensor(info[i][1]) for i in range(0, num_envs, num_agents)]
            ).to(device)
            principal_reward = principal.objective(reward) - prev_objective_val
            prev_objective_val = principal.objective(reward)
            principal_next_done = torch.zeros(args.num_parallel_games).to(
                device
            )  # for now saying principal never done

            prev_cumulative_reward = (
                torch.zeros(args.num_parallel_games, num_agents)
                if (episode_step % args.tax_period) == 0
                else ctx.cumulative_rewards[step - 1]
            )
            next_cumulative_reward = prev_cumulative_reward.to(device) + torch.tensor(
                extrinsic_reward
            ).to(device).view(
                -1, num_agents
            )  # split reward into dimensions by game
            next_cumulative_reward = next_cumulative_reward.to(device)
            ctx.cumulative_rewards[step] = next_cumulative_reward.to(device)
            ctx.a_rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                done
            ).to(device)

            ctx.p_rewards[step] = torch.tensor(principal_reward).to(device).view(-1)

            episode_step += 1

        principal_episode_rewards += torch.sum(ctx.p_rewards, 0)
        episode_rewards += torch.sum(ctx.a_rewards, 0)
        end_step = episode_step - 1

        ctx.episode_world_obs[num_updates_for_this_ep - 1] = ctx.p_obs[
            :, 0, :, :, :
        ].clone()
        if args.save_model and ctx.current_episode % args.save_model_freq == 0:
            try:
                os.mkdir(f"./saved_params_{run_name}")
            except FileExistsError:
                pass
            try:
                os.mkdir(f"./saved_params_{run_name}/ep{ctx.current_episode}")
            except FileExistsError:
                pass
            torch.save(
                ctx.a_obs,
                f"./saved_params_{run_name}/ep{ctx.current_episode}/obs_samplerun{num_updates_for_this_ep}_ep{ctx.current_episode}.pt",
            )
            torch.save(
                ctx.a_actions,
                f"./a_saved_params_{run_name}/ep{ctx.current_episode}/actions_samplerun{num_updates_for_this_ep}_ep{ctx.current_episode}.pt",
            )
            torch.save(
                ctx.a_logprobs,
                f"./saved_params_{run_name}/ep{ctx.current_episode}/logprobs_samplerun{num_updates_for_this_ep}_ep{ctx.current_episode}.pt",
            )
            torch.save(
                ctx.a_rewards,
                f"./saved_params_{run_name}/ep{ctx.current_episode}/rewards_samplerun{num_updates_for_this_ep}_ep{ctx.current_episode}.pt",
            )
            torch.save(
                ctx.a_dones,
                f"./saved_params_{run_name}/ep{ctx.current_episode}/a_dones_samplerun{num_updates_for_this_ep}_ep{ctx.current_episode}.pt",
            )
            torch.save(
                ctx.a_values,
                f"./saved_params_{run_name}/ep{ctx.current_episode}/values_samplerun{num_updates_for_this_ep}_ep{ctx.current_episode}.pt",
            )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = ctx.agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(ctx.a_rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.sampling_horizon)):
                if t == args.sampling_horizon - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - ctx.a_dones[t + 1]
                    nextvalues = ctx.a_values[t + 1]
                delta = (
                    ctx.a_rewards[t] + args.gamma * nextvalues * nextnonterminal - ctx.a_values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + ctx.a_values

        # bootstrap principal value if not done
        with torch.no_grad():
            principal_next_value = ctx.principal_agent.get_value(
                principal_next_obs, next_cumulative_reward
            ).reshape(1, -1)
            principal_advantages = torch.zeros_like(ctx.p_rewards).to(device)
            principal_lastgaelam = 0
            for t in reversed(range(args.sampling_horizon)):
                if t == args.sampling_horizon - 1:
                    principal_nextnonterminal = 1.0 - principal_next_done
                    principal_nextvalues = principal_next_value
                else:
                    principal_nextnonterminal = 1.0 - ctx.p_dones[t + 1]
                    principal_nextvalues = ctx.p_values[t + 1]
                principal_delta = (
                    ctx.p_rewards[t]
                    + args.gamma * principal_nextvalues * principal_nextnonterminal
                    - ctx.p_values[t]
                )
                principal_advantages[t] = principal_lastgaelam = (
                    principal_delta
                    + args.gamma
                    * args.gae_lambda
                    * principal_nextnonterminal
                    * principal_lastgaelam
                )
            principal_returns = principal_advantages + ctx.p_values

        # flatten the batch
        b_obs = ctx.a_obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = ctx.a_logprobs.reshape(-1)
        b_actions = ctx.a_actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = ctx.a_values.reshape(-1)

        # Optimizing the agent policy and value network
        b_inds = np.arange(len(b_obs))
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_obs), args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = ctx.agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                ctx.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ctx.agent.parameters(), args.max_grad_norm)
                ctx.optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # flatten batch for principal
        principal_b_obs = ctx.p_obs.reshape((-1,) + (144, 192, 3))
        principal_b_logprobs = ctx.p_logprobs.reshape(-1)
        b_cumulative_rewards = ctx.cumulative_rewards.reshape(
            -1, num_agents
        )  # from sampling_horizon x num_games x num_agents to (sampling_horizon*num_games) x num_agents
        principal_b_actions = ctx.p_actions.reshape((-1, 3))
        principal_b_advantages = principal_advantages.reshape(-1)
        principal_b_returns = principal_returns.reshape(-1)
        principal_b_values = ctx.p_values.reshape(-1)

        # Optimizing the principal policy and value network
        b_inds = np.arange(len(principal_b_obs))
        principal_clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(
                0, len(principal_b_obs), args.minibatch_size // num_agents
            ):  # principal has batch size num_games not num_envs(=num_games*num_agents) so divide to ensure same number of minibatches as agents
                end = start + args.minibatch_size // num_agents
                mb_inds = b_inds[start:end]

                (
                    _,
                    principal_newlogprob,
                    principal_entropy,
                    principal_newvalue,
                ) = ctx.principal_agent.get_action_and_value(
                    principal_b_obs[mb_inds],
                    b_cumulative_rewards[mb_inds],
                    principal_b_actions.long()[mb_inds],
                )
                principal_logratio = (
                    principal_newlogprob - principal_b_logprobs[mb_inds]
                )
                principal_ratio = principal_logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    principal_old_approx_kl = (-principal_logratio).mean()
                    principal_approx_kl = (
                        (principal_ratio - 1) - principal_logratio
                    ).mean()
                    principal_clipfracs += [
                        ((principal_ratio - 1.0).abs() > args.clip_coef)
                        .float()
                        .mean()
                        .item()
                    ]

                principal_mb_advantages = principal_b_advantages[mb_inds]
                if args.norm_adv:
                    principal_mb_advantages = (
                        principal_mb_advantages - principal_mb_advantages.mean()
                    ) / (principal_mb_advantages.std() + 1e-8)

                # Policy loss
                principal_pg_loss1 = -principal_mb_advantages * principal_ratio
                principal_pg_loss2 = -principal_mb_advantages * torch.clamp(
                    principal_ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                principal_pg_loss = torch.max(
                    principal_pg_loss1, principal_pg_loss2
                ).mean()

                # Value loss
                principal_newvalue = principal_newvalue.view(-1)
                if args.clip_vloss:
                    principal_v_loss_unclipped = (
                        principal_newvalue - principal_b_returns[mb_inds]
                    ) ** 2
                    principal_v_clipped = principal_b_values[mb_inds] + torch.clamp(
                        principal_newvalue - principal_b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    principal_v_loss_clipped = (
                        principal_v_clipped - principal_b_returns[mb_inds]
                    ) ** 2
                    principal_v_loss_max = torch.max(
                        principal_v_loss_unclipped, principal_v_loss_clipped
                    )
                    principal_v_loss = 0.5 * principal_v_loss_max.mean()
                else:
                    principal_v_loss = (
                        0.5
                        * (
                            (principal_newvalue - principal_b_returns[mb_inds]) ** 2
                        ).mean()
                    )

                principal_entropy_loss = principal_entropy.mean()
                principal_loss = (
                    principal_pg_loss
                    - args.ent_coef * principal_entropy_loss
                    + principal_v_loss * args.vf_coef
                )

                ctx.principal_optimizer.zero_grad()
                principal_loss.backward()
                nn.utils.clip_grad_norm_(
                    ctx.principal_agent.parameters(), args.max_grad_norm
                )
                ctx.principal_optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        principal_y_pred, principal_y_true = (
            principal_b_values.cpu().numpy(),
            principal_b_returns.cpu().numpy(),
        )
        principal_var_y = np.var(principal_y_true)
        principal_explained_var = (
            np.nan
            if principal_var_y == 0
            else 1 - np.var(principal_y_true - principal_y_pred) / principal_var_y
        )

        # one more policy update done
        num_updates_for_this_ep += 1
        print(
            f"Completed policy update {num_updates_for_this_ep} for episode {ctx.current_episode} - used steps {start_step} through {end_step}"
        )

        if num_updates_for_this_ep == ctx.num_policy_updates_per_ep:
            # episode finished

            if args.capture_video and ctx.current_episode % args.video_freq == 0:
                # currently only records first of any parallel games running but
                # this is easily changed at the point where we add to episode_world_obs
                video = torch.cat(ctx.episode_world_obs, dim=0).cpu()
                try:
                    os.mkdir(f"./videos_{run_name}")
                except FileExistsError:
                    pass
                torchvision.io.write_video(
                    f"./videos_{run_name}/episode_{ctx.current_episode}.mp4", video, fps=20
                )
                huggingface_upload.upload(f"./videos_{run_name}", run_name)
                if args.track:
                    wandb.log(
                        {
                            "video": wandb.Video(
                                f"./videos_{run_name}/episode_{ctx.current_episode}.mp4"
                            )
                        }
                    )
                os.remove(f"./videos_{run_name}/episode_{ctx.current_episode}.mp4")

            writer.add_scalar(
                "charts/learning_rate", ctx.optimizer.param_groups[0]["lr"], ctx.current_episode
            )
            writer.add_scalar("losses/value_loss", v_loss.item(), ctx.current_episode)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), ctx.current_episode)
            writer.add_scalar("losses/entropy", entropy_loss.item(), ctx.current_episode)
            writer.add_scalar(
                "losses/old_approx_kl", old_approx_kl.item(), ctx.current_episode
            )
            writer.add_scalar("losses/approx_kl", approx_kl.item(), ctx.current_episode)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), ctx.current_episode)
            writer.add_scalar(
                "losses/explained_variance", explained_var, ctx.current_episode
            )
            writer.add_scalar(
                "charts/mean_episodic_return",
                torch.mean(episode_rewards),
                ctx.current_episode,
            )
            writer.add_scalar("charts/episode", ctx.current_episode, ctx.current_episode)
            writer.add_scalar("charts/tax_frac", tax_frac, ctx.current_episode)
            mean_rewards_across_envs = {
                player_idx: 0 for player_idx in range(0, num_agents)
            }
            for idx in range(len(episode_rewards)):
                mean_rewards_across_envs[idx % num_agents] += episode_rewards[
                    idx
                ].item()
            mean_rewards_across_envs = list(
                map(
                    lambda x: x / args.num_parallel_games,
                    mean_rewards_across_envs.values(),
                )
            )

            for player_idx in range(num_agents):
                writer.add_scalar(
                    f"charts/episodic_return-player{player_idx}",
                    mean_rewards_across_envs[player_idx],
                    ctx.current_episode,
                )
            print(
                f"Finished episode {ctx.current_episode}, with {ctx.num_policy_updates_per_ep} policy updates"
            )
            print(f"Mean episodic return: {torch.mean(episode_rewards)}")
            print(f"Episode returns: {mean_rewards_across_envs}")
            print(f"Principal returns: {principal_episode_rewards.tolist()}")
            for game_id in range(args.num_parallel_games):
                writer.add_scalar(
                    f"charts/principal_return_game{game_id}",
                    principal_episode_rewards[game_id].item(),
                    ctx.current_episode,
                )
                for tax_period in range(len(tax_values)):
                    tax_step = (
                        ctx.current_episode - 1
                    ) * args.episode_length // args.tax_period + tax_period
                    for bracket in range(0, 3):
                        writer.add_scalar(
                            f"charts/tax_value_game{game_id}_bracket_{bracket+1}",
                            np.array(
                                tax_values[tax_period][f"game_{game_id}"][bracket]
                            ),
                            tax_step,
                        )

            print(
                f"Tax a_values this episode (for each period): {tax_values}, capped by multiplier {tax_frac}"
            )
            print("*******************************")

            if args.save_model and ctx.current_episode % args.save_model_freq == 0:
                try:
                    os.mkdir(f"./models_{run_name}")
                except FileExistsError:
                    pass
                torch.save(
                    ctx.agent.state_dict(),
                    f"./models_{run_name}/agent_{ctx.current_episode}.pth",
                )
                torch.save(
                    ctx.principal_agent.state_dict(),
                    f"./models_{run_name}/principal_{ctx.current_episode}.pth",
                )
                huggingface_upload.upload(f"./models_{run_name}", run_name)
                os.remove(f"./models_{run_name}/agent_{ctx.current_episode}.pth")
                os.remove(f"./models_{run_name}/principal_{ctx.current_episode}.pth")

                huggingface_upload.upload(f"./saved_params_{run_name}", run_name)
                shutil.rmtree(f"./saved_params_{run_name}/ep{ctx.current_episode}")
                print("model saved")

            # vote on principal objective
            principal_objective = vote(voting_values)
            principal.set_objective(principal_objective)

            # start a new episode:
            next_obs = torch.Tensor(envs.reset()).to(device)
            next_done = torch.zeros(num_envs).to(device)
            # no need to reset obs,actions,logprobs,etc as they have length args.sampling_horizon so will be overwritten

            ctx.current_episode += 1
            num_updates_for_this_ep = 0
            episode_step = 0
            prev_objective_val = 0
            episode_rewards = torch.zeros(num_envs).to(device)
            principal_episode_rewards = torch.zeros(args.num_parallel_games).to(device)
            tax_values = []

    envs.close()
    writer.close()




if __name__ == "__main__":
    main()
