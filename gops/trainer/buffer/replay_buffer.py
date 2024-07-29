#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Replay buffer
#  Update: 2021-03-05, Yuheng Lei: Create replay buffer


import numpy as np
import sys
import torch
from typing import Sequence, Union
from gops.utils.common_utils import set_seed
from gops.trainer.sampler.base import Experience

__all__ = ["ReplayBuffer"]


def combined_shape(length: int, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class BufferData:
    def __init__(self, max_size: int, vec_dim, obs_dim: Union[int, Sequence[int]], act_dim: Union[int, Sequence[int]], additional_info: dict):
        self.max_size = max_size
        self.vec_dim = vec_dim
        self.data = {
            "obs": np.zeros(
                combined_shape(self.max_size, obs_dim), dtype=np.float32
            ),
            "obs2": np.zeros(
                combined_shape(self.max_size, obs_dim), dtype=np.float32
            ),
            "act": np.zeros(
                combined_shape(self.max_size, act_dim), dtype=np.float32
            ),
            "rew": np.zeros(self.max_size, dtype=np.float32),
            "done": np.zeros(self.max_size, dtype=np.float32).astype(bool),
            "finished": np.zeros(self.max_size, dtype=np.float32).astype(bool),
            "logp": np.zeros(self.max_size, dtype=np.float32),

        }
        self.additional_info = additional_info
        for k, v in self.additional_info.items():
            if isinstance(v, dict):
                self.data[k] = np.zeros(
                    combined_shape(self.max_size, v["shape"]), dtype=v["dtype"]
                )
                self.data["next_" + k] = np.zeros(
                    combined_shape(self.max_size, v["shape"]), dtype=v["dtype"]
                )
            else:
                self.data[k] = v.batch(self.max_size)
                self.data["next_" + k] = v.batch(self.max_size)

        self.ptr, self.prev_ptr, self.size, = (0, 0, 0)
        self.memory_usage = self.__sizeof__()

    @property
    def cur_memory_usage(self):
        return self.ptr*self.memory_usage/self.max_size
    
    def __prev(self, idxes: np.ndarray) -> np.ndarray:       
        prev_idxes = (idxes - self.vec_dim) % self.size
        finished = self.data["finished"][prev_idxes]
        is_unconti = np.logical_or(finished, self.__is_overlapping(prev_idxes))
        prev_idxes = (prev_idxes + self.vec_dim*is_unconti) % self.size # if prev idx is not continuous, then use current idx
        return prev_idxes

    def __is_overlapping(self, prev_idxes: np.ndarray) -> np.ndarray:
        cur_idxes = prev_idxes + self.vec_dim
        temp_idxes = self.prev_ptr + (cur_idxes>= self.size) *(self.prev_ptr - self.vec_dim<=0) * self.size
        return np.logical_and(prev_idxes<= temp_idxes, temp_idxes < cur_idxes)


    
    def sample(self, idxes: np.ndarray,batch_size:int, seq_len: int =1, add_noise = False) -> dict:
        batch = {}
        assert seq_len >= 1
        if seq_len ==1:
            for k, v in self.data.items():
                if isinstance(v, np.ndarray):
                    batch[k] = torch.as_tensor(v[idxes], dtype=torch.float32)
                else:
                    batch[k] = v[idxes].array2tensor()
        else:
            idxes_list = [idxes] 
            for _ in range(seq_len - 1):
                idxes_list.append(self.__prev(idxes_list[-1]))
            idxes_list.reverse() # the last idx is the current idx
            for k, v in self.data.items():
                if k in ["obs", "obs2"]:
                    batch[k] = torch.as_tensor(np.stack([v[idx] for idx in idxes_list], axis=1), dtype=torch.float32)
                elif k in ["act"]:
                    batch[k] = torch.as_tensor(v[idxes_list[-1]], dtype=torch.float32) # only use the current action
                    
                else:
                    if isinstance(v, np.ndarray):
                        batch[k] = torch.as_tensor(v[idxes], dtype=torch.float32)
                    else:
                        batch[k] = v[idxes].array2tensor()

            if "step" in self.additional_info.keys():
                batch["step"] = torch.as_tensor(np.stack([self.data["step"][idxes] for idxes in idxes_list], axis=1), dtype=torch.float32)
                self.check_conti(batch["step"])
            if add_noise:
                obs_noise = torch.as_tensor(np.stack([self.data["noise"][idxes] for idxes in idxes_list], axis=1), dtype=torch.float32)
                noise_level_scale = torch.randint(low=0, high=3, size=(batch_size,1), dtype=torch.float32)/3
                batch["raw_obs"] = batch["obs"]
                batch["obs"] = batch["obs"] + obs_noise*noise_level_scale.unsqueeze(-1)
                # random_noise_level
                obs2_noise = torch.as_tensor(np.stack([self.data["next_noise"][idxes] for idxes in idxes_list], axis=1), dtype=torch.float32)
                batch["raw_obs2"] = batch["obs2"]
                batch["obs2"] = batch["obs2"] + obs2_noise*noise_level_scale.unsqueeze(-1)


                if isinstance(v, np.ndarray):
                    batch[k] = torch.as_tensor(np.stack([v[idxes] for idxes in idxes_list], axis=1), dtype=torch.float32)
                else:
                    batch[k] = torch.stack([v[idxes].array2tensor() for idxes in idxes_list], axis=1)
        return batch
    def check_conti(self, batch_step):
        diff_1 = batch_step[:, 1:] - batch_step[:, :-1]
        diff_2 = diff_1[:, 1:] - diff_1[:, :-1]
        assert torch.all(diff_1 <= 1) and torch.all(diff_1 >= 0) and torch.all(diff_2 >= 0), f"step is not continuous:{batch_step[torch.nonzero((diff_1 <= 1)*(diff_1 >= 0))]}" # TODO: more concise

    
    def store(self,
              exp: Experience,
              ) -> None:
        self.data["obs"][self.ptr] = exp.obs
        self.data["obs2"][self.ptr] = exp.next_obs
        self.data["act"][self.ptr] = exp.action
        self.data["rew"][self.ptr] = exp.reward
        self.data["done"][self.ptr] = exp.done 
        self.data["finished"][self.ptr] = exp.finished
        self.data["logp"][self.ptr] = exp.logp
        for k in self.additional_info.keys():
            self.data[k][self.ptr] = exp.info[k]
            self.data["next_" + k][self.ptr] = exp.next_info[k] 
        
        self.prev_ptr = self.ptr
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size) 
    
    def __sizeof__(self) -> int:
        size = sys.getsizeof(self.data)
        for key, value in self.data.items():
        # Add the size of the key and the size of the value (assuming value is a numpy array or tensor)
            if isinstance(value, np.ndarray):
                size += sys.getsizeof(key) + value.nbytes
            elif isinstance(value, torch.Tensor):
                size += sys.getsizeof(key) + value.element_size() * value.nelement()
            else:
                size += sys.getsizeof(key) + sys.getsizeof(value) # if value is not a numpy array or tensor, then use sys.getsizeof
        return size/1024**2 # return the size in MB
        

class ReplayBuffer:
    """
    Implementation of replay buffer with uniform sampling probability.
    """

    def __init__(self, index=0, **kwargs):
        set_seed(kwargs["trainer"], kwargs["seed"], index + 100)
        self.obsv_dim = kwargs["obsv_dim"]
        self.vec_env_num = kwargs.get("vector_env_num", 1)
        self.act_dim = kwargs["action_dim"]
        self.max_size = kwargs["buffer_max_size"]
        self.seq_len = kwargs.get("seq_len", 1)
        self.freeze_iteration = kwargs.get("freeze_iteration", 0)
        if self.freeze_iteration > 0:
            self.seq_len_after_freeze = self.seq_len
            self.seq_len = 1
        else:
            self.seq_len_after_freeze = self.seq_len
        self.add_noise = kwargs.get("add_noise", False)
        self.buf = BufferData(self.max_size,self.vec_env_num, self.obsv_dim, self.act_dim, kwargs["additional_info"])

    def change_mode(self):
        self.add_noise = True
        self.seq_len = self.seq_len_after_freeze

    def __len__(self):
        return self.buf.size
    
    @property
    def size(self):
        return self.buf.size
    
    @property
    def ptr(self):
        return self.buf.ptr
    
    @property
    def prev_ptr(self):
        return self.buf.prev_ptr


    def __get_RAM__(self):
        return self.buf.cur_memory_usage
    
    def store(self, sample: Experience) -> None:
        self.buf.store(sample)

    def add_batch(self, samples: list) -> None:
        list(map(lambda sample: self.store(sample), samples))

    def sample_batch(self, batch_size: int) -> dict:
        idxes = np.random.randint(0, len(self), size=batch_size)
        batch = self.buf.sample(idxes, batch_size, self.seq_len, add_noise = self.add_noise)
        return batch

    def sample_statistic(self, iteration, batch_size: int = 1024) -> dict:
        batch = self.sample_batch(batch_size)
        vx,_ = torch.sort(batch["obs"][:, 0])
        ref_points_num =4
        yref, _ = torch.sort(torch.abs(batch["obs"][:, 7 + ref_points_num])) # TODO: Hard code
        max_yref = np.max(self.buf["obs"][:, 7 + ref_points_num])
        print(f'max yref = {max_yref}')
        print(f'ref_points_num = {ref_points_num}')
        return {
            'vx': (
                f"{iteration:d},"
                f"{vx.mean():.2f},"
                f"{vx.std():.2f},"
                f"{vx[0]:.2f},"
                f"{vx[int(batch_size * 0.25)]:.2f},"
                f"{vx[int(batch_size * 0.5)]:.2f},"
                f"{vx[int(batch_size * 0.75)]:.2f},"
                f"{vx[-1]:.2f}"
            ),
            'y_ref': (
                f"{iteration:d},"
                f"{yref.mean():.2f},"
                f"{yref.std():.2f},"
                f"{yref[0]:.2f},"
                f"{yref[int(batch_size * 0.25)]:.2f},"
                f"{yref[int(batch_size * 0.5)]:.2f},"
                f"{yref[int(batch_size * 0.75)]:.2f},"
                f"{yref[-1]:.2f}"
            )
        }