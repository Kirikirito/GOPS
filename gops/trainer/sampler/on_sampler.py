#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Monte Carlo Sampler
#  Update Date: 2022-10-20, Yuheng Lei: Revise Codes
#  Update Date: 2021-03-10, Wenhan CAO: Revise Codes
#  Update Date: 2021-03-05, Wenxuan Wang: add action clip
#  Update Date: 2023-07-22, Zhilong Zheng: inherit from BaseSampler


from typing import List

import numpy as np
import torch

from gops.trainer.sampler.base import BaseSampler, Experience


class OnSampler(BaseSampler):
    def __init__(
        self, 
        sample_batch_size,
        index=0, 
        noise_params=None,
        **kwargs
    ):
        super().__init__(
            sample_batch_size,
            index, 
            noise_params,
            **kwargs
        )
        
        alg_name = kwargs["algorithm"]
        self.gamma = 0.99  #? why hard-coded?
        if self._is_vector:
            self.obs_dim = self.env.single_observation_space.shape
            self.act_dim = self.env.single_action_space.shape
        else:
            self.obs_dim = self.env.observation_space.shape
            self.act_dim = self.env.action_space.shape

        self.mb_obs = np.zeros(
            (self.num_envs, self.horizon, *self.obs_dim), dtype=np.float32
        )
        self.mb_act = np.zeros(
            (self.num_envs, self.horizon, *self.act_dim), dtype=np.float32
        )
        self.mb_rew = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
        self.mb_done = np.zeros((self.num_envs, self.horizon), dtype=np.bool_)
        self.mb_finished = np.zeros((self.num_envs, self.horizon), dtype=np.bool_)
        self.mb_tlim = np.zeros((self.num_envs, self.horizon), dtype=np.bool_)
        self.mb_logp = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
        self.need_value_flag = not (alg_name == "FHADP" or alg_name == "INFADP")
        if self.need_value_flag:
            self.gae_lambda = 0.95
            self.mb_val = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
            self.mb_adv = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
            self.mb_ret = np.zeros((self.num_envs, self.horizon), dtype=np.float32)

        self.seq_len = kwargs.get("seq_len", 1)
        self.freeze_iteration = kwargs.get("freeze_iteration", 0)
        if self.freeze_iteration > 0:
            self.seq_len_after_freeze = self.seq_len
            self.seq_len = 1
        else:
            self.seq_len_after_freeze = self.seq_len
        self.add_noise = kwargs.get("add_noise", False)
        self.mb_info = {}
        self.info_keys = kwargs["additional_info"].keys()
        for k, v in kwargs["additional_info"].items():
            self.mb_info[k] = np.zeros(
                (self.num_envs, self.horizon, *v["shape"]), dtype=v["dtype"]
            )
            self.mb_info["next_" + k] = np.zeros(
                (self.num_envs, self.horizon, *v["shape"]), dtype=v["dtype"]
            )

    def _sample(self) -> dict:
        self.ptr = np.zeros(self.num_envs, dtype=np.int32)
        self.last_ptr = np.zeros(self.num_envs, dtype=np.int32)
        for t in range(self.horizon):
            # batch_obs has shape (num_envs, obs_dim)
            if not self._is_vector:
                batch_obs = torch.from_numpy(
                    np.expand_dims(self.obs, axis=0).astype("float32")
                )
            else:
                batch_obs = torch.from_numpy(self.obs.astype("float32"))
            # interact with environment
            experiences = self._step()
            self._process_experiences(experiences, batch_obs, t)

        # wrap collected data into replay format
        mb_data = {
            "obs": torch.from_numpy(self.mb_obs.reshape(-1, *self.obs_dim)),
            "act": torch.from_numpy(self.mb_act.reshape(-1, *self.act_dim)),
            "rew": torch.from_numpy(self.mb_rew.reshape(-1)),
            "done": torch.from_numpy(self.mb_done.reshape(-1)),
            "finished": torch.from_numpy(self.mb_finished.reshape(-1)),
            "logp": torch.from_numpy(self.mb_logp.reshape(-1)),
            "time_limited": torch.from_numpy(self.mb_tlim.reshape(-1)),
        }
        if self.need_value_flag:
            mb_data.update({
                "ret": torch.from_numpy(self.mb_ret.reshape(-1)),   
                "adv": torch.from_numpy(self.mb_adv.reshape(-1)),
            })
        for k, v in self.mb_info.items():
            mb_data[k] = torch.from_numpy(v.reshape(-1, *v.shape[2:]))
        return mb_data
    def reprocess_samples(self, data):
        if self.seq_len == 1:
            mb_data = {
            "obs": torch.from_numpy(self.mb_obs.reshape(-1, *self.obs_dim)),
            "act": torch.from_numpy(self.mb_act.reshape(-1, *self.act_dim)),
            "rew": torch.from_numpy(self.mb_rew.reshape(-1)),
            "done": torch.from_numpy(self.mb_done.reshape(-1)),
            "finished": torch.from_numpy(self.mb_finished.reshape(-1)),
            "logp": torch.from_numpy(self.mb_logp.reshape(-1)),
            "time_limited": torch.from_numpy(self.mb_tlim.reshape(-1)),
        }
        else:
            obs_seq = np.stack([self.mb_obs[:, i:i+self.seq_len, ...] for i in range(self.horizon - self.seq_len + 1)], axis=1)
            obs2_seq = np.stack([self.mb_obs[:, i:i+self.seq_len, ...] for i in range(1, self.horizon - self.seq_len + 2)], axis=1)
            unconti_idx = np.where(self.mb_finished)
            for i in range(self.seq_len):
                rel_unconti_idx = np.clip(unconti_idx- i, 0, self.horizon-self.seq_len) 
                obs_seq[rel_unconti_idx, i:] = self.mb_obs[unconti_idx]
            
            mb_data = {
                "obs": torch.from_numpy(obs_seq.reshape(-1, self.seq_len, *self.obs_dim)),
                "act": torch.from_numpy(self.mb_act.reshape(-1, *self.act_dim)),
                "rew": torch.from_numpy(self.mb_rew.reshape(-1)),
                "done": torch.from_numpy(self.mb_done.reshape(-1)),
                "finished": torch.from_numpy(self.mb_finished.reshape(-1)),
                "logp": torch.from_numpy(self.mb_logp.reshape(-1)),
                "time_limited": torch.from_numpy(self.mb_tlim.reshape(-1)),
            }
        if self.need_value_flag:    
            mb_data.update({
                "ret": torch.from_numpy(self.mb_ret.reshape(-1)),   
                "adv": torch.from_numpy(self.mb_adv.reshape(-1)),
            })
        for k, v in self.mb_info.items():
            mb_data[k] = torch.from_numpy(v.reshape(-1, *v.shape[2:]))
        return mb_data

    def _prev_idx(self, idxes: np.ndarray) -> np.ndarray:
        prev_idxes = (idxes - 1)
        prev_idxes = np.maximum(prev_idxes, 0)
        finished = self.mb_finished[prev_idxes]
        prev_idxes = (prev_idxes + 1 * finished)
        return prev_idxes

    def sample_with_replay_format(self):
        return self.sample()

    def _process_experiences(
        self, 
        experiences: List[Experience],
        batch_obs: np.ndarray, 
        t: int
    ):
        if self.need_value_flag:
            value = self.networks.value(batch_obs).detach()
            self.mb_val[:, t] = value
        
        for i in np.arange(self.num_envs):
            (
                obs, 
                action, 
                reward, 
                done,
                finished, 
                info, 
                next_obs, 
                next_info, 
                logp,
            ) = experiences[i]

            (
                self.mb_obs[i, t, ...],
                self.mb_act[i, t, ...],
                self.mb_rew[i, t],
                self.mb_done[i, t],
                self.mb_finished[i, t],
                self.mb_tlim[i, t],
                self.mb_logp[i, t],
            ) = (
                obs,
                action,
                reward,
                done,
                finished,
                next_info["TimeLimit.truncated"],
                logp,
            )

            for key in self.info_keys:
                self.mb_info[key][i, t] = info[key]
                self.mb_info["next_" + key][i, t] = next_info[key]

            # calculate value target (mb_ret) & gae (mb_adv)
            if (
                finished
                or t == self.horizon - 1
            ) and self.need_value_flag:
                last_obs_expand = torch.from_numpy(
                    np.expand_dims(next_obs, axis=0).astype("float32")
                )
                est_last_value = self.networks.value(
                    last_obs_expand
                ).detach().item() * (1 - done)
                self.ptr[i] = t
                self._finish_trajs(i, est_last_value)
                self.last_ptr[i] = self.ptr[i]

    def _finish_trajs(self, env_index: int, est_last_val: float):
        # calculate value target (mb_ret) & gae (mb_adv) whenever episode is finished
        path_slice = slice(self.last_ptr[env_index] + 1, self.ptr[env_index] + 1)
        value_preds_slice = np.append(self.mb_val[env_index, path_slice], est_last_val)
        rews_slice = self.mb_rew[env_index, path_slice]
        length = len(rews_slice)
        ret = np.zeros(length)
        adv = np.zeros(length)
        gae = 0.0
        for i in reversed(range(length)):
            delta = (
                rews_slice[i]
                + self.gamma * value_preds_slice[i + 1]
                - value_preds_slice[i]
            )
            gae = delta + self.gamma * self.gae_lambda * gae
            ret[i] = gae + value_preds_slice[i]
            adv[i] = gae
        self.mb_ret[env_index, path_slice] = ret
        self.mb_adv[env_index, path_slice] = adv
