from typing import Dict, Tuple, int, str

from abc import abstractmethod

import numpy as np
import torch
from eztils.torch import set_gpu_mode, zeros, zeros_like
from tensordict import TensorDict

from research_project.utils import Config
from sen.neural.agent_architectures import FlattenedEpisodeInfo, StepInfo


# Base class for AgentBuffer and PrincipalBuffer so that we can toggle in optimize policy
class BaseBuffer:
    def __init__(self, num_envs, base_shape, key_shape_dict):
        self.num_envs = num_envs
        self.base_shape = base_shape
        self.tensordict = TensorDict.fromkeys(
            ["logprobs", "rewards", "dones", "values"],
            zeros(self.base_shape),
        )
        self.specialize_tensordict(key_shape_dict, base_shape)

    def specialize_tensordict(self, key_shape_dict):
        for key, shape in key_shape_dict.items():
            self.tensordict[key] = zeros(shape)

    def record_data(self, av: StepInfo, step):
        self.tensordict["actions"][step] = av.action
        self.tensordict["values"][step] = av.value
        self.tensordict["logprobs"][step] = av.log_prob
        self.tensordict["obs"][step] = av.obs
        self.tensordict["dones"][step] = av.done

    def record_reward(self, reward, step):
        self.tensordict["rewards"][step] = reward

    @abstractmethod
    def reshape_for_opt(self, *args, **kwargs):
        pass


class AgentBuffer(BaseBuffer):
    def __init__(self, num_envs, base_shape, obs_shape, action_shape):
        key_shape_dict = {"obs": obs_shape, "actions": action_shape}
        super().__init__(num_envs, base_shape, key_shape_dict)

    def reshape_for_opt(self, envs) -> FlattenedEpisodeInfo:
        return FlattenedEpisodeInfo(
            actions=self.tensordict["actions"].reshape(
                (-1,) + envs.single_action_space.shape
            ),
            log_probs=self.tensordict["logprobs"].reshape(-1),
            values=self.tensordict["values"].reshape(-1),
            obs=self.tensordict["obs"].reshape(
                (-1,) + envs.single_observation_space.shape
            ),
        )


class PrincipalBuffer(BaseBuffer):
    def __init__(self, num_envs, base_shape, obs_shape, action_shape, cumulative_shape):
        key_shape_dict = {"obs": obs_shape}
        super().__init__(num_envs, base_shape, key_shape_dict)

        self.tensordict["actions"] = zeros((base_shape[0], base_shape[1], action_shape))
        self.tensordict["cumulative_reward"] = zeros(
            (base_shape[0], base_shape[1], cumulative_shape)
        )

    def log_cumulative(self, cumulative_reward, step):
        self.tensordict["cumulative_reward"][step] = cumulative_reward

    def reshape_for_opt(self, envs, num_agents) -> FlattenedEpisodeInfo:
        return FlattenedEpisodeInfo(
            actions=self.tensordict["actions"].reshape((-1, 3)),
            log_probs=self.tensordict["logprobs"].reshape(-1),
            values=self.tensordict["values"].reshape(-1),
            obs=self.tensordict["obs"].reshape((-1,) + (144, 192, 3)),
            cumulative_rewards=self.tensordict["cumulative_reward"].reshape(
                -1, num_agents
            ),
        )


class BufferList:
    def __init__(
        self, agent_buffer: AgentBuffer, principal_buffer: PrincipalBuffer
    ) -> None:
        self.agent_buffer = agent_buffer
        self.principal_buffer = principal_buffer

    def record_both(self, agent_av: StepInfo, principal_av: StepInfo, step):
        self.agent_buffer.record_data(agent_av, step)
        self.principal_buffer.record_data(principal_av, step)

    def record_both_reward(self, agent_reward, principal_reward, step):
        self.agent_buffer.record_reward(agent_reward, step)
        self.principal_buffer.record_reward(principal_reward, step)
