#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: data type environment wrapper that add noise to observation
#  Update: 2022-10-27, Congsheng Zhang: create noise observation wrapper


from typing import Tuple

import gym
import torch
import numpy as np
from gym.utils import seeding
from gym.core import ObsType, ActType
from gops.env.wrapper.base import ModelWrapper
from gops.utils.gops_typing import InfoDict
from typing import Tuple, Dict

# define a sine generator with next() method
class SineGenerator:
    def __init__(self, frequency,amplitude, phase= 0.25):
        self.amplitude = amplitude
        self.frequency = frequency
        assert np.max(frequency) <= 0.5 and np.min(frequency) >= 0
        self.phase = phase + np.random.uniform(np.zeros_like(frequency), 1/frequency)

    def __iter__(self):
        return self

    def __next__(self):
        self.phase += self.frequency
        return self.amplitude * np.sin(self.phase*2*np.pi)


class NoiseData(gym.Wrapper):
    """Data type environment wrapper that add noise to observation.

    :param env: data type environment.
    :param str noise_type: distribution of noise, support Normal distribution and Uniform distribution.
    :param np.ndarray noise_data: if noise_type == "normal", noise_data means Mean and
        Standard deviation of Normal distribution. if noise_type == "uniform", noise_data means Upper
        and Lower bounds of Uniform distribution.
    """

    def __init__(self, env, noise_type: str, noise_data: list, seed= None, add_to_info=False, rel_noise_scale=False, record_step_info =True):
        super(NoiseData, self).__init__(env)
        assert noise_type in ["normal", "uniform","sine"]
        assert (
            len(noise_data) == 2
        )
        for i in range (2):
            if type(noise_data[i]) is float or type(noise_data[i]) is int or type(noise_data[i]) is np.float64:
                noise_data[i] = [noise_data[i]] * env.observation_space.shape[0]
                print("Single value noise data is detected, automatically expand to match observation space")

        self.noise_type = noise_type
        self.noise_data = np.array(noise_data, dtype=np.float32)
        if self.noise_type == "sine":
            noise_data[0] = np.array(noise_data[0], dtype=np.float32)
            noise_data[1] = np.array(noise_data[1], dtype=np.float32)
            self.sine_generator = SineGenerator(self.noise_data[0], self.noise_data[1])
        self.seed(seed)
        self.add_to_info = add_to_info
        self.rel_noise_scale = rel_noise_scale
        if self.rel_noise_scale:
            self.running_mean_delta_obs = 0
            self.prev_obs = 0
        # self.seed(1919)
        self.record_step_info = record_step_info
        self._step = 0

    def observation(self, observation):
        if self.rel_noise_scale:
            delta_obs = np.abs(observation - self.prev_obs)
            self.prev_obs = observation
            self.running_mean_delta_obs = 0.999 * self.running_mean_delta_obs + 0.001 * delta_obs
            scale = np.abs(self.running_mean_delta_obs) + 1e-6
        else:
            scale = 1
        if self.noise_type is None:
            noise = 0
        elif self.noise_type == "normal":
            noise = scale*self.np_random.normal(
                loc=self.noise_data[0], scale=self.noise_data[1]
            )
        elif self.noise_type == "uniform":
            noise = scale*self.np_random.uniform(
                low=self.noise_data[0], high=self.noise_data[1]
            )
        elif self.noise_type == "sine":
            noise = scale*next(self.sine_generator)
        if self.add_to_info:
            return observation, noise
        else:
            return observation + noise, noise


    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._step = 0
        if self.rel_noise_scale:
            self.prev_obs = obs
        obs_noised, noise = self.observation(obs)
        if self.add_to_info:
            info["noise"] = noise
            info["running_mean_obs"] = self.running_mean_delta_obs
        if self.record_step_info:
            info["step"] = self._step
            self._step += 1
        return obs_noised, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, r, d, info = self.env.step(action)
        obs_noised, noise = self.observation(obs)
        if self.add_to_info:
            info["noise"] = noise
        if self.record_step_info:
            info["step"] = self._step
            self._step += 1
        return obs_noised, r, d, info

    def seed(self, seed=None):
        seeds = self.env.seed(seed)  # NOTE: must first seed env before seeding np_random
        np_random, _ = seeding.np_random(seed)
        noise_seed = int(np_random.randint(2 ** 31))
        self.np_random, noise_seed = seeding.np_random(noise_seed)

        return seeds

    @property
    def additional_info(self):
        if not hasattr(self.env, "additional_info"):
            env_info = {}
        else:
            env_info = self.env.additional_info
        if self.add_to_info:
            noise_sample = self.observation(self.env.observation_space.sample())[1]
            env_info["noise"] = {"shape": noise_sample.shape, "dtype": noise_sample.dtype}
        if self.record_step_info:
            env_info["step"] = {"shape": (1,), "dtype": np.int32}
        return env_info



class NoiseModel(ModelWrapper):
    """Model type environment wrapper that add noise to observation.

    :param str noise_type: distribution of noise, support Normal distribution and Uniform distribution.
    :param np.ndarray noise_data: if noise_type == "normal", noise_data means Mean and
        Standard deviation of Normal distribution. if noise_type == "uniform", noise_data means Upper
        and Lower bounds of Uniform distribution.
    """

    def __init__(self, model, noise_type: str, noise_data: list):
        super(NoiseModel, self).__init__(model)
        assert noise_type in ["normal", "uniform"]
        self.noise_type = noise_type
        self.noise_data = torch.tensor(noise_data)


    def observation(self, observation):
        if self.noise_type is None:
            return observation
        elif self.noise_type == "normal":
            observation = observation + torch.normal(self.noise_data[0], self.noise_data[1])
            return observation
        elif self.noise_type == "uniform":
            return observation + torch.rand(self.noise_data[0], self.noise_data[1])


    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        done: torch.Tensor,
        info: InfoDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        next_obs, reward, next_done, next_info = super().forward(
            obs, action, done, info
        )
        next_obs = self.observation(next_obs)
        return next_obs, reward, next_done, next_info
