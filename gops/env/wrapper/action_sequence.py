#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: action repeat wrappers for data and model type environment
#  Update: 2022-11-15, Wenxuan Wang: create action repeat wrapper


from __future__ import annotations

from typing import TypeVar, Tuple, Union

import gym
import numpy as np
import torch
from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
from gops.env.wrapper.base import ModelWrapper
from gops.utils.gops_typing import InfoDict

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class ActionSeqData(gym.Wrapper):
    """Action sequence wrapper for data type environments.
    Args:
        env (gym.Env): Environment to wrap.
        seq_len (int): Number of times to repeat the action.
        sum_reward (bool): If True, the reward is the sum of the rewards in the sequence.
    """

    def __init__(self, env, seq_len: int = 1, sum_reward: bool = True):
        super(ActionSeqData, self).__init__(env)
        self.seq_len = seq_len
        self.sum_reward = sum_reward
        # action space
        self.origin_action_space = env.action_space
        self.action_dim = env.action_space.shape[0]
        low = np.repeat(env.action_space.low, seq_len)
        high = np.repeat(env.action_space.high, seq_len)
        self.action_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=env.action_space.dtype,
        )


    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        sum_r = 0
        for i in range(self.seq_len):
            # the action is rolled out in sequence, get the real action by slicing
            real_action = action[i * self.action_dim : (i + 1) * self.action_dim]
            obs, r, d, info = self.env.step(real_action)
            sum_r += r
            if d:
                break
        if not self.sum_reward:
            sum_r = r
        return obs, sum_r, d, info


class ActionSeqModel(ModelWrapper):

    def __init__(
        self, model: PythBaseModel, repeat_num: int = 1, sum_reward: bool = True
    ):
        super(ActionSeqModel, self).__init__(model)
        self.repeat_num = repeat_num
        self.sum_reward = sum_reward

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, InfoDict]:

       raise NotImplementedError