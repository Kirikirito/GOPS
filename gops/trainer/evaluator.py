#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Evaluation of trained policy
#  Update Date: 2021-05-10, Yang Guan: renew environment parameters


import numpy as np
import torch

from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_alg import create_approx_contrainer
from gops.utils.common_utils import set_seed
from functorch import jacrev, vmap


class Evaluator:
    def __init__(self, index=0, **kwargs):
        kwargs.update({
            "reward_scale": None,
            "repeat_num": None,
            "gym2gymnasium": False,
            "vector_env_num": None,
        })
        self.env = create_env(**kwargs)

        _, self.env = set_seed(kwargs["trainer"], kwargs["seed"], index + 400, self.env)

        self.networks = create_approx_contrainer(**kwargs)
        self.render = kwargs["is_render"]

        self.num_eval_episode = kwargs["num_eval_episode"]
        self.action_type = kwargs["action_type"]
        self.policy_func_name = kwargs["policy_func_name"]
        self.save_folder = kwargs["save_folder"]
        self.eval_save = kwargs.get("eval_save", True)
        self.algorithm = kwargs["algorithm"]

        self.print_time = 0
        self.print_iteration = -1

        if self.algorithm =="ACDPI":
            self.loss_lambda = kwargs["policy_lambda"]
            self.act_dim = kwargs["action_dim"]
            self.eps = kwargs["policy_eps"]
            self.min_log_std = kwargs["policy_min_log_std"]
            self.max_log_std = kwargs["policy_max_log_std"]
            self.target_multiple = kwargs["target_multiple"]
            self.auto_lambda = kwargs["auto_lambda"]
            self.lambda_learning_rate = kwargs["lambda_learning_rate"]
            self.level = kwargs["noise_level"]
            self.additional_info = kwargs["additional_info"]
            self.obs_dim = kwargs["obsv_dim"]
            self.noise_switch = kwargs["noise_switch"]
        else:
            self.loss_lambda = 0            

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
        mlp_action_list, lips_action_list = [], []
        obs, info = self.env.reset()
        done = 0
        info["TimeLimit.truncated"] = False
        while not (done or info["TimeLimit.truncated"]):
            if self.algorithm == "ACDPI":
                if self.noise_switch:
                    mean = [0] * self.obs_dim
                    std0 = [1e-6 * self.level] * self.obs_dim
                    ## [v, w, dy, dphi, dv, d_x, d_y, d_phi, d_v, d_w]
                    # std0 = [0.01 * self.level, 0.017 * self.level, 0.01 * self.level, 0.017 * self.level, 0.01 * self.level, 0.01 * self.level,
                    #        0.01 * self.level, 0.017 * self.level, 0.01 * self.level, 0.017 * self.level]
                    std = std0 #+ std1
                    noise = [np.random.normal(m, s, 1) for m, s in zip(mean, std)]
                    noise_list = [n[0].tolist() for n in noise]
                    obs_n = np.array([[x + y for x, y in zip(obs, noise_list)]])
                    obs = obs_n.tolist()[0]

                batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
                logits = self.networks.policy(batch_obs)
                pre_obs = self.networks.policy.pi_net(batch_obs)

                action_distribution_mlp = self.networks.create_action_distributions(logits)
                action_mlp = action_distribution_mlp.mode()
                action_mlp = action_mlp.detach().numpy()[0]
                mlp_action_list.append(action_mlp)

                logits_mean, logits_std = torch.chunk(logits, chunks=2, dim=-1)
                jacobi = vmap(jacrev(self.networks.policy.policy))(pre_obs).detach()
                norm = torch.norm(jacobi[:, : self.act_dim, :], 2, dim=(2)).detach()

                k_out = self.networks.K(pre_obs)
                k_value, logits_std = torch.chunk(k_out, chunks=2, dim=-1)
                logits_std = torch.clamp(
                    logits_std, self.min_log_std, self.max_log_std
                ).exp()
                mean_lips = k_value * logits_mean / (norm + self.eps)
                # mean_lips = logits_mean
                logits_lips = torch.cat((mean_lips, logits_std), dim=1)
                lips_action_distribution = self.networks.create_action_distributions(logits_lips)
                lips_action = lips_action_distribution.mode()
                action = lips_action.detach().numpy()[0]
                lips_action_list.append(action)
            else:
                batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
                logits = self.networks.policy(batch_obs)
                action_distribution = self.networks.create_action_distributions(logits)
                action = action_distribution.mode()
                action = action.detach().numpy()[0]

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

        if self.algorithm == "ACDPI" and self.auto_lambda and iteration > 5000:
            mlp_flu = self.fluctuation(mlp_action_list)
            lips_flu = self.fluctuation(lips_action_list)
            self.real_multiple = mlp_flu / lips_flu + self.eps
            self.loss_lambda = self.loss_lambda - self.lambda_learning_rate * (
                                self.real_multiple - self.target_multiple)
            self.loss_lambda = max(1e-8, min(self.loss_lambda, 1))
        else:
            self.loss_lambda = self.loss_lambda
            self.real_multiple = 0

        self.lambda_list.append(self.loss_lambda)
        self.mean_mul_list.append(self.real_multiple)

        eval_dict = {
            "reward_list": reward_list,
            "action_list": action_list,
            "obs_list": obs_list,
        }
        if self.eval_save:
            np.save(
                self.save_folder
                + "/evaluator/iter{}_ep{}".format(iteration, self.print_time),
                eval_dict,
            )
        episode_return = sum(reward_list)
        return episode_return

    def run_n_episodes(self, n, iteration):
        episode_return_list = []
        self.lambda_list = []
        self.mean_mul_list = []
        if self.algorithm == "ACDPI":
            for _ in range(n):
                episode_return_list.append(self.run_an_episode(iteration, self.render))
                self.new_lambda = np.mean(self.lambda_list)
                self.mean_mul = np.mean(self.mean_mul_list)
            return np.mean(episode_return_list),self.new_lambda , self.mean_mul
        else:
            for _ in range(n):
                episode_return_list.append(self.run_an_episode(iteration, self.render))
            return np.mean(episode_return_list)


    def run_evaluation(self, iteration):
        return self.run_n_episodes(self.num_eval_episode, iteration)

    def fluctuation(self, list):
        a_t = np.array(list[1:])
        a_t_last = np.array(list[:-1])
        self.fluctuation_rate = np.linalg.norm(a_t - a_t_last, ord=2, axis=-1).mean()
        return self.fluctuation_rate
