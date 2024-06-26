import numpy as np
import torch
import torch.nn as nn
from eztils.torch import zeros_like

from harvest_sed.buffer import BaseBuffer
from harvest_sed.logger import MLLogger
from harvest_sed.neural.agent_architectures import (
    BaseAgent,
    FlattenedEpisodeInfo,
    PrincipalAgent,
    StepInfo,
)
from harvest_sed.utils import Config, Context


def optimize_policy(
    args: Config,
    ctx: Context,
    base_agent: BaseAgent,
    logger: MLLogger,
    buffer: BaseBuffer,
    update,
):
    principal = isinstance(base_agent, PrincipalAgent)
    if principal:
        prefix = "principal_"
    else:
        prefix = ""
    with torch.no_grad():
        # TODO: find out if value with final observation is boilerplate, if so put it in context or base_agent (instead of next_obs)
        # get value of next state - after the episode, next_obs for both principal and agent will be the last obs
        next_value = base_agent.get_value(base_agent.next_obs).reshape(1, -1)
        advantages = zeros_like(buffer.tensordict["rewards"])
        lastgaelam = 0  # Generalized Advantage Estimation
        for t in reversed(range(args.sampling_horizon)):
            # nextnonterminal is a boolean flag indicating if the episode ends on the next step
            # nextvalues holds the estimated value of the next state
            if t == args.sampling_horizon - 1:  # last time step
                nextnonterminal = 1.0 - base_agent.next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - buffer.tensordict["dones"][t + 1]
                nextvalues = buffer.tensordict["values"][t + 1]
            # Compute TD error
            delta = (
                buffer.tensordict["rewards"][t]
                + args.gamma * nextvalues * nextnonterminal
                - buffer.tensordict["values"][t]
            )
            advantages[t] = lastgaelam = (
                delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            )  # update advantage for current time step

    # flatten the batch
    returns = advantages + buffer.tensordict["values"]
    returns = returns.reshape(-1)
    agent_info: FlattenedEpisodeInfo = buffer.reshape_for_opt()
    advantages = advantages.reshape(-1)
    # sets returns equal to Q(s,a) -- values gives average expected return, which we adjust with the advantage to be more accurate
    for r in range(len(returns)):
        logger.log_return(wandb_data={f"collect/{prefix}returns": returns[r].mean()})
    logger.flush_return()
    inds = np.arange(len(agent_info.obs))  # array of indices for the batch
    logger.clipfracs = []  # list to store clipping fractions
    # pass these in as args
    if (
        principal
    ):  # this deals with the few discrepancies between optimizing principal vs agent
        mb_size = args.minibatch_size // ctx.num_agents
        optimizer = ctx.principal_optimizer
    else:
        mb_size = args.minibatch_size
        optimizer = ctx.optimizer

    for epoch in range(args.update_epochs):
        np.random.shuffle(inds)  # shuffle indices for mini-batching
        for start in range(0, len(agent_info.obs), mb_size):
            end = start + mb_size
            mb_inds = inds[start:end]  # indices for minibatch
            if principal:
                new_agent_info = base_agent.get_action_and_value(
                    agent_info.obs[mb_inds],
                    agent_info.cumulative_rewards[mb_inds],
                    agent_info.actions.long()[mb_inds],
                )
            else:
                new_agent_info = base_agent.get_action_and_value(
                    agent_info.obs[mb_inds], agent_info.actions.long()[mb_inds]
                )
            logger = ctx.alg.get_policy_gradient(
                new_agent_info,
                agent_info,
                logger,
                args.norm_adv,
                advantages,
                args.clip_coef,
                mb_inds,
            )
            # Value loss
            logger = ctx.alg.get_value_fn_gradient(
                new_agent_info,
                args.clip_vloss,
                args.clip_coef,
                returns,
                mb_inds,
                agent_info,
                logger,
            )

            logger.entropy_loss = new_agent_info.entropy.mean()
            loss = (
                logger.pg_loss
                - args.ent_coef * logger.entropy_loss
                + logger.v_loss * args.vf_coef
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(base_agent.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.target_kl is not None:
            if logger.approx_kl > args.target_kl:
                break

        logger.log(
            wandb_data={
                f"opt/epoch": epoch + args.update_epochs * (update - 1),
                f"opt/{prefix}pg_loss": logger.pg_loss.item(),
                f"opt/{prefix}v_loss": logger.v_loss.item(),
                f"opt/{prefix}entropy_loss": logger.entropy_loss.item(),
                f"opt/{prefix}approx_kl": logger.approx_kl,
                f"opt/{prefix}clipfrac": np.mean(logger.clipfracs),
            },
            flush=False,
        )
