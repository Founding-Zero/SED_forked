import numpy as np
import torch
import torch.nn as nn
from eztils.torch import zeros_like

from research_project.buffer import AgentBuffer, BaseBuffer, BufferList, PrincipalBuffer
from research_project.logger import Logger
from research_project.utils import Config, Context
from sen.neural.agent_architectures import (
    BaseAgent,
    FlattenedEpisodeInfo,
    PrincipalAgent,
    StepInfo,
)


def optimize_policy(
    args: Config,
    ctx: Context,
    base_agent: BaseAgent,
    logger: Logger,
    buffer: BaseBuffer,
):
    principal = isinstance(base_agent, PrincipalAgent)
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
        agent_info: FlattenedEpisodeInfo = buffer.reshape_for_opt(ctx.num_agents)
        advantages = advantages.reshape(-1)
        # sets returns equal to Q(s,a) -- values gives average expected return, which we adjust with the advantage to be more accurate
        returns = (advantages + buffer.tensordict["values"]).reshape(-1)

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
                get_policy_gradient(
                    new_agent_info,
                    agent_info,
                    logger,
                    args.norm_adv,
                    advantages,
                    args.clip_coef,
                    mb_inds,
                )
                # Value loss
                get_loss_fn_gradient(
                    new_agent_info,
                    args.clip_vloss,
                    args.clip_coef,
                    returns,
                    mb_inds,
                    agent_info,
                    logger,
                )
                logger.entropy_loss = agent_info.entropy.mean()
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

        save_explained_var(logger, agent_info, returns, principal)


def get_loss_fn_gradient(
    new_agent_info: StepInfo,
    clip_vloss,
    clip_coef,
    returns,
    inds,
    agent_info: FlattenedEpisodeInfo,
    logger: Logger,
):
    new_agent_info.value = new_agent_info.value.view(-1)
    if clip_vloss:  # check if clipped
        v_loss_unclipped = (new_agent_info.value - returns[inds]) ** 2
        v_clipped = agent_info.values[inds] + torch.clamp(
            new_agent_info.value - agent_info.values[inds],
            -clip_coef,
            clip_coef,
        )
        v_loss_clipped = (v_clipped - returns[inds]) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        logger.v_loss = 0.5 * v_loss_max.mean()
    else:
        logger.v_loss = 0.5 * ((new_agent_info.value - returns[inds]) ** 2).mean()


def get_policy_gradient(
    new_agent_info: StepInfo,
    agent_info: FlattenedEpisodeInfo,
    logger: Logger,
    norm_adv: bool,
    advantages,
    clip_coef,
    inds,
):
    logratio = (
        new_agent_info.log_prob - agent_info.actions[inds]
    )  # ratio betwen new and old policy
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        logger.old_approx_kl = (-logratio).mean()
        logger.approx_kl = ((ratio - 1) - logratio).mean()
        logger.clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

    mb_advantages = advantages[inds]  # advantages for the minibatch
    if norm_adv:  # normalize if specified
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
            mb_advantages.std() + 1e-8
        )

    # Policy loss (normal and clipped) Note these are negative, and that we use torch.max()
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(
        ratio, 1 - clip_coef, 1 + clip_coef
    )  # compute
    logger.pg_loss = torch.max(pg_loss1, pg_loss2).mean()


def save_explained_var(
    logger: Logger, agent_info: FlattenedEpisodeInfo, returns, principal
):
    y_pred, y_true = agent_info.values.cpu().numpy(), returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    if principal:
        logger.principal_explained_var = explained_var
    else:
        logger.explained_var = explained_var
