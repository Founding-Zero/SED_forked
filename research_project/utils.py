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
import random
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import utils as gym_utils
from meltingpot import substrate
from meltingpot.examples.gym import utils
import torch
from ml_collections import config_dict
from pettingzoo import utils as pettingzoo_utils
from pettingzoo.utils import wrappers
import supersuit as ss
import gymnasium as gym
import torch.optim as optim
from tensordict import TensorDict
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


def set_up_envs(env, frames, parallel_games):
    env.render_mode = "rgb_array"
    env = ss.observation_lambda_v0(env, lambda x, _: x["RGB"], lambda s: s["RGB"])
    env = ss.frame_stack_v1(env, frames)
    env = ss.agent_indicator_v0(env, type_only=False)
    env = pettingzoo_env_to_vec_env_v1(env)
    envs = sb3_concat_vec_envs_v1(  # need our own as need reset to pass up world obs and nearby in info
        env, num_vec_envs=parallel_games
    )
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    return envs


def set_agent_preferencts(num_agents):
    voting_values = np.random.uniform(size=[num_agents])
    selfishness = np.random.uniform(size=[num_agents])
    trust = np.random.uniform(size=[num_agents])
    return voting_values, selfishness, trust


@dataclasses.dataclass
class Config:
    exp_name: str = "apple_picking_game"  # the name of this experiment
    seed: int = 1  # seed of the experiment
    torch_deterministic: bool = (
        True  # if toggled, `torch.backends.cudnn.deterministic=False`
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
    tax_period: int = (
        50  # the number of timesteps tax periods last (at end of period tax vals updated and taxes applied)
    )
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
    clip_vloss: bool = (
        True  # Toggles whether or not to use a clipped loss for the value function, as per the paper.
    )
    ent_coef: float = 0.01  # coefficient of the entropy
    vf_coef: float = 0.5  # coefficient of the value function
    max_grad_norm: float = 0.5  # the maximum norm for the gradient clipping
    target_kl: float = None  # the target KL divergence threshold


class Context:
    def __init__(self, args: Config, num_agents, num_envs, envs, device):
        self.voting_values = np.random.uniform(size=[num_agents])
        self.selfishness = np.random.uniform(size=[num_agents])
        self.trust = np.random.uniform(size=[num_agents])

        self.agent = Agent(envs).to(device)
        self.principal_agent = PrincipalAgent(num_agents).to(device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=args.learning_rate, eps=args.adam_eps
        )
        self.principal_optimizer = optim.Adam(
            self.agent.parameters(), lr=args.learning_rate, eps=args.adam_eps
        )

        # ALGO Logic: Storage setup
        agent_info = {
            "logprobs": torch.zeros((args.sampling_horizon, num_envs)).to(device),
            "rewards": torch.zeros((args.sampling_horizon, num_envs)).to(device),
            "dones ": torch.zeros((args.sampling_horizon, num_envs)).to(device),
            "values ": torch.zeros((args.sampling_horizon, num_envs)).to(device),
            "obs": torch.zeros(
                (args.sampling_horizon, num_envs) + envs.single_observation_space.shape
            ).to(device),
            "actions": torch.zeros(
                (args.sampling_horizon, num_envs) + envs.single_action_space.shape
            ).to(device),
        }
        agent_tensordict = TensorDict(
            agent_info, batch_size=[args.sampling_horizon, num_envs]
        )
        principal_info = {
            "principal_logprobs": torch.zeros(
                (args.sampling_horizon, args.num_parallel_games)
            ).to(device),
            "principal_rewards": torch.zeros(
                (args.sampling_horizon, args.num_parallel_games)
            ).to(device),
            "principal_dones": torch.zeros(
                (args.sampling_horizon, args.num_parallel_games)
            ).to(device),
            "principal_values": torch.zeros(
                (args.sampling_horizon, args.num_parallel_games)
            ).to(device),
            "principal_obs": torch.zeros(
                (args.sampling_horizon, args.num_parallel_games) + (144, 192, 3)
            ).to(device),
            "cumulative_rewards": torch.zeros(
                (args.sampling_horizon, args.num_parallel_games, num_agents)
            ).to(device),
            "principal_actions": torch.zeros(
                (args.sampling_horizon, args.num_parallel_games, 3)
            ).to(device),
        }
        principal_tensordict = TensorDict(
            principal_info, batch_size=[args.sampling_horizon, args.num_parallel_games]
        )
        self.a_logprobs = agent_tensordict["logprobs"]
        self.a_rewards = agent_tensordict["rewards"]
        self.a_dones = agent_tensordict["dones"]
        self.a_values = agent_tensordict["values"]
        self.a_obs = (agent_tensordict["obs"],)
        self.a_actions = agent_tensordict["actions"]

        self.p_logprobs = principal_tensordict["principal_logprobs"]
        self.p_rewards = principal_tensordict["principal_rewards"]
        self.p_dones = principal_tensordict["principal_dones"]
        self.p_values = principal_tensordict["principal_values"]
        self.p_obs = principal_tensordict["principal_obs"]
        self.p_actions = principal_tensordict["principal_actions"]
        
        self.cumulative_rewards = principal_tensordict["cumulative_rewards"]

        self.next_obs = torch.Tensor(envs.reset()).to(device)
        self.next_done = torch.zeros(num_envs).to(device)
        self.next_cumulative_reward = torch.zeros(
            args.num_parallel_games, num_agents
        ).to(device)

        self.principal_next_obs = torch.stack(
            [
                torch.Tensor(envs.reset_infos[i][1])
                for i in range(0, num_envs, num_agents)
            ]
        ).to(device)
        self.principal_next_done = torch.zeros(args.num_parallel_games).to(device)

        self.num_policy_updates_per_ep = args.episode_length // args.sampling_horizon
        self.num_policy_updates_total = (
            args.num_episodes * self.num_policy_updates_per_ep
        )
        self.num_updates_for_this_ep = 0
        self.current_episode = 1
        self.episode_step = 0
        self.episode_rewards = torch.zeros(num_envs).to(device)
        self.principal_episode_rewards = torch.zeros(args.num_parallel_games).to(device)

        self.prev_objective_val = 0
        self.tax_values = []
        self.tax_frac = 1

        # fill this with sampling horizon chunks for recording if needed
        self.episode_world_obs = [0] * (
            args.episode_length // args.sampling_horizonng_horizon
        )
    


def create_env(args, env_config, num_players):
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
