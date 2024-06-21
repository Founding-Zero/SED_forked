from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from eztils.torch import zeros
from torch.distributions.categorical import Categorical


@dataclass
class StepInfo:
    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    obs: torch.Tensor
    done: torch.Tensor = None


@dataclass
class FlattenedEpisodeInfo:
    actions: torch.Tensor
    log_probs: torch.Tensor
    entropy: torch.Tensor
    values: torch.Tensor
    obs: torch.Tensor
    done: torch.Tensor
    cumulative_rewards: torch.Tensor = None


class BaseAgent(ABC, nn.Module):
    def __init__(self):
        super().__init()
        self.next_obs = None
        self.next_done = None

    @abstractmethod
    def get_value(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_action_and_value(self, *args, **kwargs):
        pass

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer


class Agent(BaseAgent):
    def __init__(self, envs, num_envs):
        super().__init__()
        self.network = nn.Sequential(
            self.layer_init(
                nn.Conv2d(envs.single_observation_space.shape[2], 32, 8, stride=4)
            ),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            self.layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = self.layer_init(
            nn.Linear(512, envs.single_action_space.n), std=0.01
        )
        self.critic = self.layer_init(nn.Linear(512, 1), std=1)

        # will be needed for optimization
        self.next_obs = torch.Tensor(envs.reset())
        self.next_done = zeros(num_envs)

    def get_value(self, x):
        return self.critic(self.network(x.permute((0, 3, 1, 2))))

    def get_action_and_value(self, x, action=None, *args, **kwargs):
        hidden = self.network(x.permute((0, 3, 1, 2)))
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return StepInfo(
            action=action,
            log_prob=probs.log_prob(action),
            entropy=probs.entropy(),
            value=self.critic(hidden).flatten(),
            obs=x,
            done=None,
        )


class PrincipalAgent(BaseAgent):
    def __init__(self, num_agents, envs, num_envs, parallel_games):
        super().__init__()
        self.conv_net = nn.Sequential(
            self.layer_init(nn.Conv2d(3, 32, 8, stride=4)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            self.layer_init(nn.Linear(64 * 14 * 20, 512)),
            nn.ReLU(),
        )
        self.fully_connected = nn.Sequential(
            self.layer_init(nn.Linear(512 + num_agents, 512)),
            nn.ReLU(),
            self.layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
        )

        self.actor_head1 = self.layer_init(nn.Linear(512, 12), std=0.01)
        self.actor_head2 = self.layer_init(nn.Linear(512, 12), std=0.01)
        self.actor_head3 = self.layer_init(nn.Linear(512, 12), std=0.01)
        self.critic = self.layer_init(nn.Linear(512, 1), std=1)

        # will be needed for optimization
        self.next_obs = torch.stack(
            [
                torch.Tensor(envs.reset_infos[i][1])
                for i in range(0, num_envs, num_agents)
            ]
        )
        self.next_done = zeros(parallel_games)
        self.next_cumulative_reward = zeros(parallel_games, num_agents)

    def get_value(self, world_obs):
        world_obs = world_obs.clone()
        world_obs /= 255.0
        conv_out = self.conv_net(world_obs.permute((0, 3, 1, 2)))
        with_rewards = torch.cat(
            (conv_out, self.next_cumulative_reward), dim=1
        )  # shape num_games x (512+num_agents)
        hidden = self.fully_connected(with_rewards)
        return self.critic(hidden)

    def get_action_and_value(
        self, world_obs, cumulative_reward, action=None, *args, **kwargs
    ):
        world_obs = world_obs.clone()
        world_obs /= 255.0
        conv_out = self.conv_net(world_obs.permute((0, 3, 1, 2)))
        with_rewards = torch.cat(
            (conv_out, cumulative_reward), dim=1
        )  # shape num_games x (512+num_agents)
        hidden = self.fully_connected(with_rewards)
        logits1 = self.actor_head1(hidden)
        logits2 = self.actor_head2(hidden)
        logits3 = self.actor_head3(hidden)
        probs1 = Categorical(logits=logits1)
        probs2 = Categorical(logits=logits2)
        probs3 = Categorical(logits=logits3)
        if action is None:
            action = torch.stack(
                [probs1.sample(), probs2.sample(), probs3.sample()], dim=1
            )
        log_prob = (
            probs1.log_prob(action[:, 0])
            + probs2.log_prob(action[:, 1])
            + probs3.log_prob(action[:, 2])
        )
        entropy = probs1.entropy() + probs2.entropy() + probs3.entropy()
        return StepInfo(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            value=self.critic(hidden).flatten(),
            obs=world_obs,
        )
