# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PettingZoo interface to meltingpot environments."""
import copy
import dataclasses
import functools
import os
import random
import warnings

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import utils as gym_utils
from meltingpot import substrate
from meltingpot.examples.gym import utils
from ml_collections import config_dict
from pettingzoo import utils as pettingzoo_utils
from pettingzoo.utils import wrappers
from tensordict import TensorDict

from research_project.Logger import Logger
from sen.neural.agent_architectures import Agent, PrincipalAgent
from sen.principal import Principal
from sen.vector_constructors import pettingzoo_env_to_vec_env_v1, sb3_concat_vec_envs_v1

PLAYER_STR_FORMAT = "player_{index}"
MAX_CYCLES = 5000


def parallel_env(env_config, max_cycles=MAX_CYCLES, principal=None):
    return _ParallelEnv(env_config, max_cycles, principal)


def raw_env(env_config, max_cycles=MAX_CYCLES):
    return pettingzoo_utils.parallel_to_aec_wrapper(
        parallel_env(env_config, max_cycles)
    )


def env(env_config, max_cycles=MAX_CYCLES):
    aec_env = raw_env(env_config, max_cycles)
    aec_env = wrappers.AssertOutOfBoundsWrapper(aec_env)
    aec_env = wrappers.OrderEnforcingWrapper(aec_env)
    return aec_env


def timestep_to_observations(timestep):
    gym_observations = {}
    for index, observation in enumerate(timestep.observation):
        gym_observations[PLAYER_STR_FORMAT.format(index=index)] = {
            key: value for key, value in observation.items() if "WORLD." not in key
        }
    nearby_observations = {}
    for index in range(len(timestep.observation)):
        nearby = timestep.observation[index]["NEARBY"]
        nearby[index] = 0  # players shouldn't count themselves as nearby
        nearby_observations[PLAYER_STR_FORMAT.format(index=index)] = nearby
    return gym_observations, nearby_observations, timestep.observation[0]["WORLD.RGB"]


class _MeltingPotPettingZooEnv(pettingzoo_utils.ParallelEnv):
    """An adapter between Melting Pot substrates and PettingZoo's ParallelEnv."""

    def __init__(self, env_config, max_cycles, principal=None):
        self.env_config = config_dict.ConfigDict(env_config)
        self.max_cycles = max_cycles
        if principal is None:
            self._env = substrate.build_from_config(
                self.env_config, roles=self.env_config.default_player_roles
            )
        else:
            from sen.principal.substrate import build_principal_from_config

            self._env = build_principal_from_config(
                self.env_config,
                roles=self.env_config.default_player_roles,
                principal=principal,
            )
        self._num_players = len(self._env.observation_spec())
        self.possible_agents = [
            PLAYER_STR_FORMAT.format(index=index) for index in range(self._num_players)
        ]
        observation_space = utils.remove_world_observations_from_space(
            utils.spec_to_space(self._env.observation_spec()[0])
        )
        self.observation_space = functools.lru_cache(maxsize=None)(
            lambda agent_id: observation_space
        )
        action_space = utils.spec_to_space(self._env.action_spec()[0])
        self.action_space = functools.lru_cache(maxsize=None)(
            lambda agent_id: action_space
        )
        self.state_space = utils.spec_to_space(
            self._env.observation_spec()[0]["WORLD.RGB"]
        )

    def state(self):
        return self._env.observation()

    def reset(self, seed=None, options=None):
        """See base class."""
        timestep = self._env.reset()
        self.agents = self.possible_agents[:]
        self.num_cycles = 0
        observations, nearby_obs, world_obs = timestep_to_observations(timestep)

        return observations, {
            agent: ({}, world_obs, nearby_obs[agent]) for agent in self.agents
        }

    def step(self, action):
        """See base class."""
        actions = [action[agent] for agent in self.agents]
        timestep = self._env.step(actions)
        rewards = {
            agent: timestep.reward[index] for index, agent in enumerate(self.agents)
        }
        self.num_cycles += 1
        done = timestep.last() or self.num_cycles >= self.max_cycles
        dones = {agent: done for agent in self.agents}
        observations, nearby_obs, world_obs = timestep_to_observations(timestep)
        infos = {agent: ({}, world_obs, nearby_obs[agent]) for agent in self.agents}
        # infos = {agent: {} for agent in self.agents}

        if done:
            self.agents = []
        return observations, rewards, dones, dones, infos

    def close(self):
        """See base class."""
        self._env.close()

    def render(self, mode="not human", filename=None):
        print(len(self.state()))
        rgb_arr = self.state()[0]["WORLD.RGB"]
        if mode == "human":
            plt.cla()
            plt.imshow(rgb_arr, interpolation="nearest")
            if filename is None:
                plt.show(block=False)
            else:
                plt.savefig(filename)
            return None
        return rgb_arr


class _ParallelEnv(_MeltingPotPettingZooEnv, gym_utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, env_config, max_cycles, principal=None):
        gym_utils.EzPickle.__init__(self, env_config, max_cycles)
        _MeltingPotPettingZooEnv.__init__(self, env_config, max_cycles, principal)


@dataclasses.dataclass
class Config:
    exp_name: str = "apple_picking_game"  # the name of this experiment
    seed: int = 1  # seed of the experiment
    torch_deterministic: bool = (
        True  # if toggled,  torch.backends.cudnn.deterministic=False`
    )
    cuda: bool = True  # if toggled, cuda will be enabled by default
    track: bool = (
        False  # if toggled, this experiment will be tracked with Weights and Biases
    )
    wandb_project_name: str = "apple-picking-game"  # the wandb's project name
    wandb_entity: str = None  # the entity (team) of wandb's project
    capture_video: bool = True  # whether to capture videos of the agent performances
    video_freq: int = 20  # capture video every how many episodes?
    save_model: bool = True  # whether to save model parameters
    save_model_freq: int = 100  # save model parameters every how many episodes?
    learning_rate: float = 2.5e-4  # the learning rate of the optimizer
    adam_eps: float = 1e-5  # eps value for the optimizer
    num_parallel_games: int = 1  # the number of parallel game environments
    num_frames: int = 4  # the number of game frames to stack together
    num_episodes: int = 100000  # the number of steps in an episode
    episode_length: int = 1000  # the number of steps in an episode
    tax_annealment_proportion: float = (
        0.02  # proportion of episodes over which to linearly anneal tax cap multiplier
    )
    sampling_horizon: int = (
        200  # the number of timesteps between policy update iterations
    )
    tax_period: int = 50  # the number of timesteps tax periods last (at end of period tax vals updated and taxes applied)
    anneal_tax: bool = (
        True  # Toggle tax cap annealing over an initial proportion of episodes
    )
    anneal_lr: bool = (
        True  # Toggle learning rate annealing for policy and value networks
    )
    gamma: float = 0.99  # the discount factor gamma
    gae_lambda: float = 0.95  # the lambda for the general advantage estimation
    minibatch_size: int = 128  # size of minibatches when training policy network
    update_epochs: int = 4  # the K epochs to update the policy
    norm_adv: bool = True  # Toggles advantages normalization
    clip_coef: float = 0.2  # the surrogate clipping coefficient
    clip_vloss: bool = True  # Toggles whether or not to use a clipped loss for the value function, as per the paper.
    ent_coef: float = 0.01  # coefficient of the entropy
    vf_coef: float = 0.5  # coefficient of the value function
    max_grad_norm: float = 0.5  # the maximum norm for the gradient clipping
    target_kl: float = None  # the target KL divergence threshold


class Context:
    def __init__(
        self, args: Config, num_agents, num_envs, envs, device, principal: Principal
    ):
        self.num_agents = num_agents
        self.num_envs = num_envs
        self.voting_values = np.random.uniform(size=[self.num_agents])
        self.selfishness = np.random.uniform(size=[self.num_agents])
        self.trust = np.random.uniform(size=[self.num_agents])

        self.device = device
        self.agent = Agent(envs).to(
            self.device
        )  # a new init function outside that you just pass to the context
        self.principal_agent = PrincipalAgent(self.num_agents).to(self.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=args.learning_rate, eps=args.adam_eps
        )
        self.principal_optimizer = optim.Adam(
            self.agent.parameters(), lr=args.learning_rate, eps=args.adam_eps
        )
        self.principal = principal

        # ALGO Logic: Storage setup
        self.agent_tensordict = TensorDict.fromkeys(
            ["logprobs", "rewards", "dones", "values"],
            torch.zeros((args.sampling_horizon, num_envs)).to(self.device),
        )
        self.agent_tensordict["obs"] = torch.zeros(
            (args.sampling_horizon, num_envs) + envs.single_observation_space.shape
        ).to(self.device)

        self.agent_tensordict["actions"] = torch.zeros(
            (args.sampling_horizon, num_envs) + envs.single_action_space.shape
        ).to(self.device)

        # maybe use torchrl replaybuffer https://pytorch.org/rl/stable/tutorials/getting-started-3.html?
        # probably use tensorclass
        #  https://github.com/pytorch/tensordict?tab=readme-ov-file#tensorclass
        self.principal_tensordict = TensorDict.fromkeys(
            ["logprobs", "rewards", "dones", "values"],
            torch.zeros((args.sampling_horizon, args.num_parallel_games)).to(
                self.device
            ),
        )
        self.principal_tensordict["obs"] = torch.zeros(
            (args.sampling_horizon, args.num_parallel_games) + (144, 192, 3)
        ).to(self.device)
        self.principal_tensordict["actions"] = torch.zeros(
            (args.sampling_horizon, args.num_parallel_games, 3)
        ).to(self.device)
        self.principal_tensordict["cumulative_rewards"] = torch.zeros(
            (args.sampling_horizon, args.num_parallel_games, self.num_agents)
        ).to(
            self.device
        )  # Not positive, but this might not belong in principal_tensordict

        self.next_obs = torch.Tensor(envs.reset()).to(self.device)
        self.next_done = torch.zeros(num_envs).to(self.device)
        self.next_cumulative_reward = torch.zeros(
            args.num_parallel_games, self.num_agents
        ).to(self.device)

        self.principal_next_obs = torch.stack(
            [
                torch.Tensor(envs.reset_infos[i][1])
                for i in range(0, num_envs, self.num_agents)
            ]
        ).to(self.device)
        self.principal_next_done = torch.zeros(args.num_parallel_games).to(self.device)
        self.principal_advantages = torch.zeros_like(
            self.principal_tensordict["principal_rewards"]
        ).to(self.device)

        self.num_policy_updates_per_ep = args.episode_length // args.sampling_horizon
        self.num_policy_updates_total = (
            args.num_episodes * self.num_policy_updates_per_ep
        )
        self.num_updates_for_this_ep = 0
        self.current_episode = 1
        self.episode_step = 0
        self.episode_rewards = torch.zeros(num_envs).to(self.device)
        self.principal_episode_rewards = torch.zeros(args.num_parallel_games).to(
            self.device
        )

        self.prev_objective_val = 0
        self.tax_values = []
        self.tax_frac = 1
        self.next_obs = torch.Tensor(envs.reset()).to(self.device)
        self.next_done = torch.zeros(self.num_envs).to(self.device)
        # fill this with sampling horizon chunks for recording if needed
        self.episode_world_obs = [0] * (
            args.episode_length // args.sampling_horizonng_horizon
        )

    def new_episode(self, envs, objective, parallel_games):
        # no need to reset obs,actions,logprobs,etc as they have length args.sampling_horizon so will be overwritten
        self.principal.set_objective(objective)
        self.next_obs = torch.Tensor(envs.reset()).to(self.device)
        self.next_done = torch.zeros(self.num_envs).to(self.device)
        self.current_episode += 1
        self.num_updates_for_this_ep = 0
        self.episode_step = 0
        self.prev_objective_val = 0
        self.episode_rewards = torch.zeros(self.num_envs).to(self.device)
        self.principal_episode_rewards = torch.zeros(parallel_games).to(self.device)
        self.tax_values = []


def set_agent_preferences(num_agents):
    voting_values = np.random.uniform(size=[num_agents])
    selfishness = np.random.uniform(size=[num_agents])
    trust = np.random.uniform(size=[num_agents])
    return voting_values, selfishness, trust


def create_envs(args: Config, env_config, num_players):
    principal = Principal(num_players, args.num_parallel_games, "egalitarian")

    env = utils.parallel_env(
        max_cycles=args.sampling_horizon, env_config=env_config, principal=principal
    )
    num_agents = env.max_num_agents
    num_envs = args.num_parallel_games * num_agents
    env.render_mode = "rgb_array"

    env = ss.observation_lambda_v0(env, lambda x, _: x["RGB"], lambda s: s["RGB"])
    env = ss.frame_stack_v1(env, args.num_frames)
    env = ss.agent_indicator_v0(env, type_only=False)
    env = pettingzoo_env_to_vec_env_v1(env)
    envs = sb3_concat_vec_envs_v1(  # need our own as need reset to pass up world obs and nearby in info
        env, num_vec_envs=args.num_parallel_games
    )

    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"
    return num_agents, num_envs, envs, principal


def anneal_lr(update, total_updates, args: Config):
    if not args.anneal_lr:
        return args.learning_rate
    frac = 1.0 - (update - 1.0) / total_updates
    lrnow = frac * args.learning_rate
    return lrnow


def anneal_tax_cap(args: Config, current_episode, curr_tax):
    if args.anneal_tax:
        tax_frac = 0.1 + 0.9 * min(
            (current_episode - 1.0)
            / (args.num_episodes * args.tax_annealment_proportion),
            1,
        )
        return tax_frac
    return curr_tax


def collect_data_for_policy_update(args: Config, ctx: Context, envs):
    start_step = ctx.episode_step
    for step in range(0, args.sampling_horizon):
        if ctx.next_obs.shape[3] != 19:
            warnings.warn(
                "hardcoded value of 12 RGB channels - check RBG/indicator channel division here"
            )
        num_rgb_channels = 12
        # we only divide the 4 stack frames x 3 RGB channels - NOT the agent indicators
        ctx.next_obs[:, :, :, :num_rgb_channels] /= 255.0
        ctx.agent_tensordict["obs"][step] = ctx.next_obs
        ctx.agent_tensordict["dones"][step] = ctx.next_done
        ctx.principal_tensordict["obs"][step] = ctx.principal_next_obs
        ctx.principal_tensordict["dones"][step] = ctx.principal_next_done

        with torch.no_grad():
            action, logprob, _, value = ctx.agent.get_action_and_value(ctx.next_obs)
            ctx.agent_tensordict["values"][step] = value.flatten()
        ctx.agent_tensordict["actions"][step] = action
        ctx.agent_tensordict["logprobs"][step] = logprob

        with torch.no_grad():
            (
                principal_action,
                principal_logprob,
                _,
                principal_value,
            ) = ctx.principal_agent.get_action_and_value(
                ctx.principal_next_obs, ctx.next_cumulative_reward
            )
            ctx.principal_tensordict["values"][step] = principal_value.flatten()

        if ctx.episode_step % args.tax_period == 0:
            # this `principal_action` is the one that was fed cumulative reward of last step of previous tax period
            # so it is acting on the full wealth accumulated last tax period and an observation of the last frame
            ctx.principal_tensordict["actions"][step] = principal_action
            ctx.principal_tensordict["logprobs"][step] = principal_logprob
            ctx.principal.update_tax_vals(principal_action)
            ctx.tax_values.append(copy.deepcopy(ctx.principal.tax_vals))
        else:
            ctx.principal_tensordict["actions"][step] = torch.tensor([11] * 3)
            ctx.principal_tensordict["actions"][step] = torch.full(
                (args.num_parallel_games, 3), 11
            )
            ctx.principal_tensordict["logprobs"][step] = torch.zeros(
                args.num_parallel_games
            )

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
        ctx.next_obs, extrinsic_reward, done, info = envs.step(action.cpu().numpy())
        ctx.principal.report_reward(extrinsic_reward)

        # mix personal and nearby a_rewards
        intrinsic_reward = np.zeros_like(extrinsic_reward)
        nearby = torch.stack(
            [torch.Tensor(info[i][2]) for i in range(0, ctx.num_envs)]
        ).to(ctx.device)
        for game_id in range(args.num_parallel_games):
            for player_id in range(ctx.num_agents):
                env_id = player_id + game_id * ctx.num_agents
                w = ctx.selfishness[player_id]
                #! TODO fix nearby_reward = sum(nearby[env_id] * game_reward)
                nearby_reward = 0
                intrinsic_reward[env_id] = (
                    w * extrinsic_reward[env_id] + (1 - w) * nearby_reward
                )

        # make sure tax is applied after extrinsic reward is used for intrinsic reward calculation
        if (ctx.episode_step + 1) % args.tax_period == 0:
            # last step of tax period
            taxes = ctx.principal.end_of_tax_period()
            extrinsic_reward -= ctx.tax_frac * np.array(list(taxes.values())).flatten()

        reward = np.zeros_like(extrinsic_reward)
        for env_id in range(len(reward)):
            player_id = env_id % ctx.num_agents
            v = ctx.trust[player_id]
            reward[env_id] = (
                v * extrinsic_reward[env_id] + (1 - v) * intrinsic_reward[env_id]
            )

        ctx.principal_next_obs = torch.stack(
            [torch.Tensor(info[i][1]) for i in range(0, ctx.num_envs, ctx.num_agents)]
        ).to(ctx.device)
        principal_reward = ctx.principal.objective(reward) - prev_objective_val
        prev_objective_val = ctx.principal.objective(reward)
        ctx.principal_next_done = torch.zeros(args.num_parallel_games).to(
            ctx.device
        )  # for now saying principal never done

        prev_cumulative_reward = (
            torch.zeros(args.num_parallel_games, ctx.num_agents)
            if (ctx.episode_step % args.tax_period) == 0
            else ctx.principal_tensordict["cumulative_rewards"][step - 1]
        )
        next_cumulative_reward = prev_cumulative_reward.to(ctx.device) + torch.tensor(
            extrinsic_reward
        ).to(ctx.device).view(
            -1, ctx.num_agents
        )  # split reward into dimensions by game
        next_cumulative_reward = next_cumulative_reward.to(ctx.device)
        ctx.principal_tensordict["cumulative_rewards"][
            step
        ] = next_cumulative_reward.to(ctx.device)
        ctx.agent_tensordict["rewards"][step] = (
            torch.tensor(reward).to(ctx.device).view(-1)
        )
        ctx.next_obs, ctx.next_done = torch.Tensor(ctx.next_obs).to(
            ctx.device
        ), torch.Tensor(done).to(ctx.device)

        ctx.principal_tensordict["rewards"][step] = (
            torch.tensor(principal_reward).to(ctx.device).view(-1)
        )

        ctx.episode_step += 1
    ctx.principal_episode_rewards += torch.sum(ctx.principal_tensordict["rewards"], 0)
    ctx.episode_rewards += torch.sum(ctx.agent_tensordict["rewards"], 0)
    ctx.episode_world_obs[ctx.num_updates_for_this_ep - 1] = ctx.principal_tensordict[
        "obs"
    ][:, 0, :, :, :].clone()
    return start_step, ctx.episode_step


#! TODO this needs to be refactored
def save_params(ctx: Context, run_name):
    try:
        os.mkdir(f"./saved_params_{run_name}")
    except FileExistsError:
        pass
    try:
        os.mkdir(f"./saved_params_{run_name}/ep{ctx.current_episode}")
    except FileExistsError:
        pass
    # Refactor
    params = ["obs", "actions", "logprobs", "rewards", "dones", "values"]
    for param in params:
        torch.save(
            ctx.agent_tensordict[param],
            f"./saved_params_{run_name}/ep{ctx.current_episode}/{param}_samplerun{ctx.num_updates_for_this_ep}_ep{ctx.current_episode}.pt",
        )


def optimize_policy(args: Config, ctx: Context, envs, logger: Logger):
    with torch.no_grad():
        next_value = ctx.agent.get_value(ctx.next_obs).reshape(1, -1)
        advantages = torch.zeros_like(ctx.agent_tensordict["rewards"]).to(ctx.device)
        lastgaelam = 0
        for t in reversed(range(args.sampling_horizon)):
            if t == args.sampling_horizon - 1:
                nextnonterminal = 1.0 - ctx.next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - ctx.agent_tensordict["dones"][t + 1]
                nextvalues = ctx.agent_tensordict["values"][t + 1]
            delta = (
                ctx.agent_tensordict["rewards"][t]
                + args.gamma * nextvalues * nextnonterminal
                - ctx.agent_tensordict["values"][t]
            )
            advantages[t] = lastgaelam = (
                delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + ctx.agent_tensordict["values"]

        # bootstrap ctx.principal value if not done
        with torch.no_grad():
            principal_next_value = ctx.principal_agent.get_value(
                ctx.principal_next_obs, ctx.next_cumulative_reward
            ).reshape(1, -1)
            ctx.principal_advantages = torch.zeros_like(
                ctx.principal_tensordict["rewards"]
            ).to(ctx.device)
            principal_lastgaelam = 0
            for t in reversed(range(args.sampling_horizon)):
                if t == args.sampling_horizon - 1:
                    principal_nextnonterminal = 1.0 - ctx.principal_next_done
                    principal_nextvalues = principal_next_value
                else:
                    principal_nextnonterminal = (
                        1.0 - ctx.principal_tensordict["dones"][t + 1]
                    )
                    principal_nextvalues = ctx.principal_tensordict["values"][t + 1]
                principal_delta = (
                    ctx.principal_tensordict["rewards"][t]
                    + args.gamma * principal_nextvalues * principal_nextnonterminal
                    - ctx.principal_tensordict["values"][t]
                )
                ctx.principal_advantages[t] = principal_lastgaelam = (
                    principal_delta
                    + args.gamma
                    * args.gae_lambda
                    * principal_nextnonterminal
                    * principal_lastgaelam
                )
            principal_returns = (
                ctx.principal_advantages + ctx.principal_tensordict["values"]
            )

        # flatten the batch
        b_obs = ctx.agent_tensordict["obs"].reshape(
            (-1,) + envs.single_observation_space.shape
        )
        b_logprobs = ctx.agent_tensordict["logprobs"].reshape(-1)
        b_actions = ctx.agent_tensordict["actions"].reshape(
            (-1,) + envs.single_action_space.shape
        )
        b_advantages = advantages.reshape(-1)
        ctx.b_returns = returns.reshape(-1)
        ctx.b_values = ctx.agent_tensordict["values"].reshape(-1)

        # Optimizing the agent policy and value network
        b_inds = np.arange(len(b_obs))
        logger.clipfracs = []
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
                    logger.old_approx_kl = (-logratio).mean()
                    logger.approx_kl = ((ratio - 1) - logratio).mean()
                    logger.clipfracs += [
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
                logger.pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - ctx.b_returns[mb_inds]) ** 2
                    v_clipped = ctx.b_values[mb_inds] + torch.clamp(
                        newvalue - ctx.b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - ctx.b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    logger.v_loss = 0.5 * v_loss_max.mean()
                else:
                    logger.v_loss = (
                        0.5 * ((newvalue - ctx.b_returns[mb_inds]) ** 2).mean()
                    )

                logger.entropy_loss = entropy.mean()
                loss = (
                    logger.pg_loss
                    - args.ent_coef * logger.entropy_loss
                    + logger.v_loss * args.vf_coef
                )

                ctx.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ctx.agent.parameters(), args.max_grad_norm)
                ctx.optimizer.step()

            if args.target_kl is not None:
                if logger.approx_kl > args.target_kl:
                    break


# try to make just a single optimize, and then pass in as a param if it's opt. policy or principal
def optimize_principal(
    args: Config, ctx: Context, start_step, end_step, logger: Logger
):
    # flatten batch for ctx.principal
    principal_b_obs = ctx.principal_tensordict["obs"].reshape((-1,) + (144, 192, 3))
    principal_b_logprobs = ctx.principal_tensordict["logprobs"].reshape(-1)
    b_cumulative_rewards = ctx.principal_tensordict["cumulative_rewards"].reshape(
        -1, ctx.num_agents
    )  # from sampling_horizon x num_games x num_agents to (sampling_horizon*num_games) x num_agents
    principal_b_actions = ctx.principal_tensordict["actions"].reshape((-1, 3))
    principal_b_advantages = ctx.principal_advantages.reshape(-1)
    principal_b_returns = ctx.principal_returns.reshape(-1)
    principal_b_values = ctx.principal_tensordict["values"].reshape(-1)

    # Optimizing the ctx.principal policy and value network
    b_inds = np.arange(len(principal_b_obs))
    principal_clipfracs = []
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(
            0, len(principal_b_obs), args.minibatch_size // ctx.num_agents
        ):  # ctx.principal has batch size num_games not num_envs(=num_games*num_agents) so divide to ensure same number of minibatches as agents
            end = start + args.minibatch_size // ctx.num_agents
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
            principal_logratio = principal_newlogprob - principal_b_logprobs[mb_inds]
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
            principal_pg_loss = torch.max(principal_pg_loss1, principal_pg_loss2).mean()

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
                    * ((principal_newvalue - principal_b_returns[mb_inds]) ** 2).mean()
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

    y_pred, y_true = ctx.b_values.cpu().numpy(), ctx.b_returns.cpu().numpy()
    var_y = np.var(y_true)
    logger.explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

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
    ctx.num_updates_for_this_ep += 1
    print(
        f"Completed policy update {ctx.num_updates_for_this_ep} for episode {ctx.current_episode} - used steps {start_step} through {end_step}"
    )
