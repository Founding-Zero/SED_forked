import copy
import warnings

import numpy as np
import torch
from eztils.torch import zeros

from harvest_sed.buffer import *
from harvest_sed.logger import MLLogger
from harvest_sed.utils import Config, Context, get_flush


def collect_data_for_policy_update(
    args: Config, ctx: Context, envs, logger: MLLogger, update
):
    start_step = ctx.episode_step

    for step in range(0, args.sampling_horizon):
        flush = get_flush(step, args.flush_interval)
        if ctx.agent.next_obs.shape[3] != 19:
            warnings.warn(
                "hardcoded value of 12 RGB channels - check RBG/indicator channel division here"
            )
        num_rgb_channels = 12
        # we only divide the 4 stack frames x 3 RGB channels - NOT the agent indicators
        # Store observations and boolean flags about end of episode
        ctx.agent.next_obs[:, :, :, :num_rgb_channels] /= 255.0
        # gathers and logs the action, logprob, and value for the current observation
        with torch.no_grad():
            agent_info = ctx.agent.get_action_and_value(ctx.agent.next_obs)
            principal_info = ctx.principal_agent.get_action_and_value(
                ctx.principal_agent.next_obs, ctx.principal_agent.next_cumulative_reward
            )
            principal_info.done = ctx.principal_agent.next_done
            agent_info.done = ctx.agent.next_done

        # The Principal chooses an action at the start of a new tax period
        if ctx.episode_step % args.tax_period == 0:
            # this `principal_action` is the one that was fed cumulative total_agent_reward of last step of previous tax period
            # so it is acting on the full wealth accumulated last tax period and an observation of the last frame
            ctx.buffer_list.record_both(agent_info, principal_info, step)
            ctx.principal.update_tax_vals(principal_info.action)
            ctx.tax_values.append(copy.deepcopy(ctx.principal.tax_vals))
        else:
            principal_info.action = torch.tensor([11] * 3)
            principal_info.action = torch.full((args.num_parallel_games, 3), 11)
            principal_info.log_prob = zeros(args.num_parallel_games)
            ctx.buffer_list.record_both(agent_info, principal_info, step)
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
        # Step with the agents action, gather related info, update extrinsic total_agent_reward of agents
        ctx.agent.next_obs, extrinsic_reward, done, info = envs.step(
            agent_info.action.cpu().numpy()
        )
        # immediately tax and redistribute reward according to tax rate
        taxes = ctx.principal.tax()
        extrinsic_reward -= ctx.tax_frac * np.array(list(taxes.values())).flatten()
        # Increment player_wealth based on taxed reward
        ctx.principal.report_reward(extrinsic_reward)

        # Calculate socially influenced reward
        socially_influenced_reward = np.zeros_like(extrinsic_reward)
        for game_id in range(args.num_parallel_games):
            total_wealth = 0
            # get average wealth
            for player_id in range(ctx.num_agents):
                env_id = player_id + game_id * ctx.num_agents
                total_wealth += extrinsic_reward[env_id]
            avg_wealth = total_wealth / ctx.num_agents
            # weight personal reward and average communal reward with selfishness values
            for player_id in range(ctx.num_agents):
                env_id = player_id + game_id * ctx.num_agents
                s = ctx.selfishness[player_id]
                socially_influenced_reward[env_id] = (
                    s * extrinsic_reward[env_id] + (1 - s) * avg_wealth
                )
        # update principal observation and total_agent_reward

        principal_reward = (
            ctx.principal.objective(extrinsic_reward) - ctx.prev_objective_val
        )
        prev_cumulative_reward = (
            zeros(args.num_parallel_games, ctx.num_agents)
            if (ctx.episode_step % args.tax_period) == 0
            else ctx.principal_buffer.tensordict["cumulative_rewards"][step - 1]
        ).to(ctx.device)
        ctx.principal_agent.next_cumulative_reward = (
            prev_cumulative_reward
            + torch.tensor(extrinsic_reward).to(ctx.device).view(-1, ctx.num_agents)
        )  # split total_agent_reward into dimensions by game

        # log cumulative reward for principal and normal reward for both
        ctx.principal_buffer.log_cumulative(
            ctx.principal_agent.next_cumulative_reward, step
        )
        ctx.buffer_list.record_both_reward(
            agent_reward=torch.tensor(socially_influenced_reward)
            .to(ctx.device)
            .view(-1),
            principal_reward=torch.tensor(principal_reward).to(ctx.device).view(-1),
            step=step,
        )
        logger.log(
            wandb_data={
                "collect/step": ctx.episode_step
                + ((update - 1) * args.sampling_horizon),
                "collect/extrinsic_reward": extrinsic_reward.mean(),
                "collect/socially_influenced_reward": socially_influenced_reward.mean(),
                "collect/principal_reward": principal_reward,
                "collect/principal_cumulative_reward": ctx.principal_agent.next_cumulative_reward.mean().item(),
            },
            flush=False,
        )

        ctx.prev_objective_val = ctx.principal.objective(extrinsic_reward)

        # Update observations and done flags
        ctx.principal_agent.next_done = zeros(
            args.num_parallel_games
        )  # for now saying principal never done
        ctx.principal_agent.next_obs = torch.stack(
            [torch.Tensor(info[i][1]) for i in range(0, ctx.num_envs, ctx.num_agents)]
        ).to(ctx.device)
        ctx.agent.next_obs, ctx.agent.next_done = torch.Tensor(ctx.agent.next_obs).to(
            ctx.device
        ), torch.Tensor(done).to(ctx.device)
        # step episode
        ctx.episode_step += 1
    # These are only used for logging
    ctx.principal_episode_rewards += torch.sum(
        ctx.principal_buffer.tensordict["rewards"], 0
    )
    ctx.episode_rewards += torch.sum(ctx.agent_buffer.tensordict["rewards"], 0)
    ctx.episode_world_obs[
        ctx.num_updates_for_this_ep - 1
    ] = ctx.principal_buffer.tensordict["obs"][:, 0, :, :, :].clone()
    logger.log(
        wandb_data={
            "episode_eval/step": update,
            "episode_eval/principal_episode_rewards": ctx.principal_episode_rewards.mean().item(),
            "episode_eval/avg_agent_episode_rewards": ctx.episode_rewards.mean().item(),
        },
        flush=True,
    )
    return start_step, ctx.episode_step
