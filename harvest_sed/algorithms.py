from abc import abstractmethod

import numpy as np
import torch

from harvest_sed.neural.agent_architectures import FlattenedEpisodeInfo, StepInfo


class BaseAlgorithm:
    @abstractmethod
    def get_policy_gradient(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_value_fn_gradient(self, *args, **kwargs):
        pass

    def get_explained_var(
        self, logger, agent_info: FlattenedEpisodeInfo, returns, principal
    ):
        y_pred, y_true = agent_info.values.cpu().numpy(), returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if principal:
            logger.principal_explained_var = explained_var
        else:
            logger.explained_var = explained_var

        return logger


class PPO(BaseAlgorithm):
    def get_policy_gradient(
        self,
        new_agent_info: StepInfo,
        agent_info: FlattenedEpisodeInfo,
        logger,
        norm_adv: bool,
        advantages,
        clip_coef,
        inds,
    ):
        logratio = (
            new_agent_info.log_prob - agent_info.log_probs[inds]
        )  # ratio betwen new and old policy
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            logger.old_approx_kl = (-logratio).mean()
            logger.approx_kl = ((ratio - 1) - logratio).mean()
            logger.clipfracs += [
                ((ratio - 1.0).abs() > clip_coef).float().mean().item()
            ]

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
        return logger

    def get_value_fn_gradient(
        self,
        new_agent_info: StepInfo,
        clip_vloss,
        clip_coef,
        returns,
        inds,
        agent_info: FlattenedEpisodeInfo,
        logger,
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
        return logger


class AlgorithmFactory:
    @staticmethod
    def get_alg(alg: str) -> BaseAlgorithm:
        if alg == "ppo":
            return PPO()
