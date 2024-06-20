from typing import Dict, Tuple, int, str

import numpy as np
import torch
from eztils.torch import set_gpu_mode, zeros, zeros_like
from tensordict import TensorDict

from research_project.utils import Config


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


class AgentBuffer(BaseBuffer):
    def __init__(self, num_envs, base_shape, obs_shape, action_shape):
        key_shape_dict = {"obs": obs_shape, "actions": action_shape}
        super().__init__(num_envs, base_shape, key_shape_dict)


class PrincipalBuffer(BaseBuffer):
    def __init__(self, num_envs, base_shape, obs_shape, action_shape, cumulative_shape):
        key_shape_dict = {"obs": obs_shape}
        super().__init__(num_envs, base_shape, key_shape_dict)
        self.tensordict["actions"] = zeros((base_shape[0], base_shape[1], action_shape))
        self.tensordict["cumulative_reward"] = zeros(
            (base_shape[0], base_shape[1], cumulative_shape)
        )
