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
import dataclasses
import functools
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import supersuit as ss
import torch
import torch.optim as optim
from eztils.torch import zeros
from gymnasium import utils as gym_utils
from meltingpot import substrate
from meltingpot1.examples.gym import utils

# from meltingpot.examples.gym import utils
from ml_collections import config_dict
from pettingzoo import utils as pettingzoo_utils
from pettingzoo.utils import wrappers

from harvest_sed.algorithms import AlgorithmFactory, BaseAlgorithm
from harvest_sed.buffer import AgentBuffer, BufferList, PrincipalBuffer
from harvest_sed.neural.agent_architectures import Agent, PrincipalAgent
from harvest_sed.principal import Principal
from harvest_sed.principal.vote import VotingFactory, VotingMechanism
from harvest_sed.vector_constructors import (
    pettingzoo_env_to_vec_env_v1,
    sb3_concat_vec_envs_v1,
)

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
    # nearby_observations = {}
    # for index in range(len(timestep.observation)):
    #     nearby = timestep.observation[index]["NEARBY"]
    #     nearby[index] = 0  # players shouldn't count themselves as nearby
    #     nearby_observations[PLAYER_STR_FORMAT.format(index=index)] = nearby
    return (
        gym_observations,
        timestep.observation[0]["WORLD.RGB"],
    )  # , nearby_observations


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
            from harvest_sed.principal.substrate import build_principal_from_config

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
        # took out nearby obs
        observations, world_obs = timestep_to_observations(timestep)
        # took this out from return: nearby_obs[agent]
        return observations, {agent: ({}, world_obs) for agent in self.agents}

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
        # deleted nearby_obs
        observations, world_obs = timestep_to_observations(timestep)
        # took this out from return: , nearby_obs[agent]
        infos = {agent: ({}, world_obs) for agent in self.agents}
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
    cuda: bool = True  # if toggled, cuda will be enabled by default
    track: bool = (
        False  # if toggled, this experiment will be tracked with Weights and Biases
    )
    wandb_project_name: str = "apple-picking-game"  # the wandb's project name
    wandb_entity: str = None  # the entity (team) of wandb's project
    log_locally: bool = False
    log_file: str = None
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
    LLM: bool = False
    flush_interval: int = 1
    voting_type: str = "simple_mean"
    selfishness_dist: str = "selfish"
    algorithm: str = "ppo"


class Context:
    def __init__(
        self,
        args: Config,
        num_agents,
        num_envs,
        envs,
        device,
        principal: Principal,
        agent: Agent,
        principal_agent: PrincipalAgent,
        agent_buffer: AgentBuffer,
        principal_buffer: PrincipalBuffer,
        buffer_list: BufferList,
        selfishness,
        alg: BaseAlgorithm,
    ):
        self.num_agents = num_agents
        self.num_envs = num_envs
        self.selfishness = selfishness
        self.device = device
        self.agent: Agent = agent
        self.principal_agent: PrincipalAgent = principal_agent

        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=args.learning_rate, eps=args.adam_eps
        )
        self.principal_optimizer = optim.Adam(
            self.agent.parameters(), lr=args.learning_rate, eps=args.adam_eps
        )
        self.principal = principal
        self.alg = alg
        # ALGO Logic: Storage setup
        self.agent_buffer = agent_buffer
        self.principal_buffer = principal_buffer
        self.buffer_list = buffer_list

        self.num_policy_updates_per_ep = args.episode_length // args.sampling_horizon
        self.num_policy_updates_total = (
            args.num_episodes * self.num_policy_updates_per_ep
        )
        self.num_updates_for_this_ep = 0
        self.current_episode = 1
        self.episode_step = 0
        self.episode_rewards = zeros(num_envs)
        self.principal_episode_rewards = zeros(args.num_parallel_games)

        self.prev_objective_val = 0
        self.tax_values = []
        self.tax_frac = 1
        self.next_obs = torch.Tensor(envs.reset())
        self.next_done = zeros(self.num_envs)
        # fill this with sampling horizon chunks for recording if needed
        self.episode_world_obs = [0] * (args.episode_length // args.sampling_horizon)

    def new_episode(self, envs, objective, parallel_games):
        # no need to reset obs,actions,logprobs,etc as they have length args.sampling_horizon so will be overwritten
        self.principal.set_objective(objective)
        self.next_obs = torch.Tensor(envs.reset())
        self.next_done = zeros(self.num_envs)
        self.current_episode += 1
        self.num_updates_for_this_ep = 0
        self.episode_step = 0
        self.prev_objective_val = 0
        self.episode_rewards = zeros(self.num_envs)
        self.principal_episode_rewards = zeros(parallel_games)
        self.tax_values = []


class SelfishnessFactory:
    @staticmethod
    def get_selfishness(selfishness_dist: str, num_agents):
        if selfishness_dist == "selfish":
            return np.ones(shape=[num_agents])
        elif selfishness_dist == "generous":
            return np.zeros(shape=[num_agents])


def set_agent_selfishness(num_agents, selfishness_dist):
    return SelfishnessFactory.get_selfishness(selfishness_dist, num_agents)


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


def save_params(ctx: Context, run_name):
    try:
        os.mkdir(f"./saved_params_{run_name}")
    except FileExistsError:
        pass
    try:
        os.mkdir(f"./saved_params_{run_name}/ep{ctx.current_episode}")
    except FileExistsError:
        pass
    params = ["obs", "actions", "logprobs", "rewards", "dones", "values"]
    for param in params:
        torch.save(
            ctx.agent_tensordict[param],
            f"./saved_params_{run_name}/ep{ctx.current_episode}/{param}_samplerun{ctx.num_updates_for_this_ep}_ep{ctx.current_episode}.pt",
        )


def get_flush(step, flush_interval):
    if step // flush_interval == 0:
        return True
    return False


def create_envs(args: Config, env_config, principal):
    env = parallel_env(
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
    return num_agents, num_envs, envs


def set_context(args: Config, device):
    env_name = "commons_harvest__open"
    env_config = substrate.get_config(env_name)

    num_players = len(env_config.default_player_roles)
    principal = Principal(num_players, args.num_parallel_games)

    num_agents, num_envs, envs = create_envs(args, env_config, principal)
    selfishness = set_agent_selfishness(num_agents, args.selfishness_dist)
    voting_mechanism: VotingMechanism = VotingFactory.get_voting_mechanism(
        args.voting_type
    )
    principal.set_objective(voting_mechanism.vote_on_p(selfishness))

    alg: BaseAlgorithm = AlgorithmFactory.get_alg(args.algorithm)
    agent = Agent(envs, num_envs)
    principal_agent = PrincipalAgent(
        num_agents, envs, num_envs, args.num_parallel_games
    )
    agent_buffer = AgentBuffer(
        num_envs,
        base_shape=(args.sampling_horizon, num_envs),
        obs_shape=envs.single_observation_space.shape,
        action_shape=envs.single_action_space.shape,
        envs=envs,
    )
    principal_buffer = PrincipalBuffer(
        num_envs,
        base_shape=(args.sampling_horizon, args.num_parallel_games),
        obs_shape=(144, 192, 3),
        action_shape=3,
        cumulative_shape=num_agents,
        num_agents=num_agents,
    )
    buffer_list = BufferList(agent_buffer, principal_buffer)
    ctx = Context(
        args=args,
        num_agents=num_agents,
        num_envs=num_envs,
        envs=envs,
        device=device,
        principal=principal,
        agent=agent,
        principal_agent=principal_agent,
        agent_buffer=agent_buffer,
        principal_buffer=principal_buffer,
        buffer_list=buffer_list,
        selfishness=selfishness,
        alg=alg,
    )
    return ctx, envs
