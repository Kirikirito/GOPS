#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Evaluation of trained policy
#  Update Date: 2021-05-10, Yang Guan: renew environment parameters


import numpy as np
import torch

from collections import deque
from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_alg import create_approx_contrainer
from gops.utils.common_utils import set_seed
try:
    from exprunner.utils.default import default_sim_time
except ImportError:
    default_sim_time = {}

class ObsBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque([],maxlen=buffer_size)
        self.buffer_size = buffer_size
    def add(self, obs):
        if not self.is_full():
            while not self.is_full():
                self.buffer.append(obs)
        else:
            self.buffer.append(obs)
    def get(self):
        if self.buffer_size == 1:
            return self.buffer[0]
        else:
            return torch.stack(list(self.buffer), dim=1)
    def is_full(self):
        return len(self.buffer) == self.buffer_size
    def clear(self):
        self.buffer.clear()

class Evaluator:
    def __init__(self, index=0, **kwargs):
        kwargs.update({
            "reward_scale": None,
            "repeat_num": None,
            "act_seq_len": 1,
            "gym2gymnasium": False,
            "vector_env_num": None,
            "obs_noise_type": None,
        })
        self.env = create_env(**kwargs)
        self.env_id = kwargs["env_id"]
        self.obs_buffer = ObsBuffer(kwargs.get("seq_len", 1)) # for backward compatibility

        _, self.env = set_seed(kwargs["trainer"], kwargs["seed"], index + 400, self.env)

        self.networks = create_approx_contrainer(**kwargs)
        self.use_gpu = kwargs["use_gpu"]
        if self.use_gpu:
            self.networks = self.networks.cuda()
        self.device = self.networks.device
        self.networks.eval()
        self.render = kwargs["is_render"]

        self.num_eval_episode = kwargs["num_eval_episode"]
        self.action_type = kwargs["action_type"]
        self.policy_func_name = kwargs["policy_func_name"]
        self.save_folder = kwargs["save_folder"]
        self.eval_save = kwargs.get("eval_save", True)

        self.print_time = 0
        self.print_iteration = -1

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)

    def run_an_episode(self, iteration, render=True):
        if self.print_iteration != iteration:
            self.print_iteration = iteration
            self.print_time = 0
        else:
            self.print_time += 1
        obs_list = []
        action_list = []
        reward_list = []
        obs, info = self.env.reset()
        done = 0
        info["TimeLimit.truncated"] = False
        self.obs_buffer.clear()
        while not (done or info["TimeLimit.truncated"]):
            batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
            self.obs_buffer.add(batch_obs)
            logits = self.networks.policy(self.obs_buffer.get().to(self.device))
            action_distribution = self.networks.create_action_distributions(logits)
            action = action_distribution.mode()
            action = action.detach().cpu().numpy()[0]
            next_obs, reward, done, next_info = self.env.step(action)
            obs_list.append(obs)
            action_list.append(action)
            obs = next_obs
            info = next_info
            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            # Draw environment animation
            if render:
                self.env.render()
            reward_list.append(reward)
        eval_dict = {
            "reward_list": reward_list,
            "action_list": np.array(action_list),
            "obs_list": obs_list,
        }
        if self.eval_save:
            np.save(
                self.save_folder
                + "/evaluator/iter{}_ep{}".format(iteration, self.print_time),
                eval_dict,
            )
        episode_return = sum(reward_list)
        eval_result = {}
        eval_result["episode_return"] = episode_return
        eval_result.update(self._cal_afr(eval_dict))
        eval_result.update(self._cal_sdv(eval_dict))
        eval_result.update(self._cal_mwf(eval_dict))
        

        return eval_result

    def run_n_episodes(self, n, iteration):
        eval_list = [self.run_an_episode(iteration, self.render) for _ in range(n)]
        avg_eval_dict = {
         k: np.mean([d[k] for d in eval_list]) for k in eval_list[0].keys()
        }
        return avg_eval_dict

    def run_evaluation(self, iteration):
        return self.run_n_episodes(self.num_eval_episode, iteration)
    
    
    def _cal_sdv(self, eval_dict):
        action_seq = eval_dict["action_list"]
        action_seq = np.array(action_seq)
        daction_seq = action_seq[1:] - action_seq[:-1]
        ddaction_seq = daction_seq[1:] - daction_seq[:-1]

        sdv = np.linalg.norm(ddaction_seq, axis=1).mean()
        sdv_std = np.linalg.norm(ddaction_seq, axis=1).std()

        eval_result = {
            "sdv": sdv,
            "sdv_std": sdv_std,
        }
        return eval_result
    

    
    def _cal_mwf(self, eval_dict):
        total_mwf = 0
        env_id = self.env_id
        if default_sim_time.get(env_id) is None:
            # print(f"env_id {env_id} is not in default_sim_time, use default value 0.005")
            T = 0.005
        else:
            T = default_sim_time[env_id]

        action_seq = eval_dict["action_list"]
        N = len(action_seq)
        dim = action_seq.shape[1]
        mwf = 0
        top = 0
        freq_list = []
        amp_list = []
        for i in range(dim):
            y = action_seq[:,i]
            yf = np.fft.fft(y)
            xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
            # remove first top points
            top = 0
            yf[:top] = 0
            weighted_yf = np.sum(xf[top:N//2]*np.abs(yf[top:N//2]))/(np.sum(np.abs(yf[top:N//2]))+1e-8)
            avg_amp = np.sum(np.abs(yf[top:N//2]))/(N//2-top)
            mwf += weighted_yf
            freq_list.append(xf[top:N//2])
            amp_list.append(np.abs(yf[top:N//2]))
            total_mwf += mwf
            
        mwf = total_mwf/dim

        eval_result = {
            "mwf": mwf,
        }
        return eval_result
    
    
    def _cal_afr(self, eval_dict):
        action_seq = eval_dict["action_list"]
        action_seq = np.array(action_seq)
        daction_seq = action_seq[1:] - action_seq[:-1]
        afr = np.linalg.norm(daction_seq, axis=1).mean()
        afr_std = np.linalg.norm(daction_seq, axis=1).std()

        eval_result = {
            "afr": afr,
            "afr_std": afr_std,
        }
        return eval_result

