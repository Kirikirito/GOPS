#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Evaluator for IDSim when training
#  Update Date: 2023-11-22, Guojian Zhan: create this file
from typing import Dict, List, Tuple

import numpy as np
import torch
import json
import pathlib
from gops.trainer.evaluator import Evaluator
from gops.env.env_gen_ocp.resources.idsim_tags import idsim_tb_tags_dict, reward_tags
from functorch import jacrev, vmap


class EvalResult:
    def __init__(self):
        # training info
        self.iteration: int = None
        # scenario info
        self.map_path: str = None
        self.map_id: str = None
        self.seed: int = None
        self.traffic_seed: int = None
        self.warmup_time: float = None
        self.save_folder: str = None
        self.ego_id: str = None
        self.ego_route: Tuple = None
        # evaluation info
        self.done_info: Dict[str, int] = {}
        self.reward_info: Dict[str, List[float]] = {k: [] for k in reward_tags}
        self.obs_list: List[np.ndarray] = []
        self.action_list: List[np.ndarray] = []
        self.reward_list: List[np.ndarray] = []

class IdsimTrainEvaluator(Evaluator):
    def __init__(self, index=0, **kwargs):
        super().__init__(index, **kwargs)
        self.max_iteration = kwargs["max_iteration"]
        self.env_seed_rng = np.random.default_rng(kwargs["seed"])
        self.algorithm = kwargs["algorithm"]

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



    def run_an_episode(self, iteration, render=True):
        if self.print_iteration != iteration:
            self.print_iteration = iteration
            self.print_time = 0
        else:
            self.print_time += 1


        idsim_tb_eval_dict = {key: 0. for key in idsim_tb_tags_dict.keys()}
        env_seed = self.env_seed_rng.integers(0, 2**30)
        obs, info = self.env.reset(seed=env_seed)

        eval_result = EvalResult()

        env_context = self.env.engine.context
        warmup_time = env_context.simulation_time
        vehicle = env_context.vehicle
        eval_result.iteration = iteration
        eval_result.map_path = str(env_context.scenario.root)
        eval_result.map_id = str(env_context.scenario_id)
        eval_result.seed = int(env_seed)
        eval_result.traffic_seed = env_context.traffic_seed
        eval_result.warmup_time = warmup_time
        eval_result.save_folder = str(self.save_folder)
        eval_result.ego_id = str(vehicle.id)
        eval_result.ego_route = vehicle.route


        mlp_action_list, lips_action_list = [], []

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
            eval_result.obs_list.append(obs)
            eval_result.action_list.append(action)
            obs = next_obs
            info = next_info
            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            for eval_key in idsim_tb_eval_dict.keys():
                if eval_key in info.keys():
                    idsim_tb_eval_dict[eval_key] += info[eval_key]
                if eval_key in info["reward_details"].keys():
                    idsim_tb_eval_dict[eval_key] += info["reward_details"][eval_key]
            # Draw environment animation
            if render:
                self.env.render()
            eval_result.reward_list.append(reward)
        for k, v in idsim_tb_eval_dict.items():
            if k.startswith("done"):
                eval_result.done_info[k] = v
        episode_return = sum(eval_result.reward_list)
        idsim_tb_eval_dict["total_avg_return"] = episode_return
        if iteration > 0*self.max_iteration / 5:
            self.save_eval_scenario(eval_result)

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

        return idsim_tb_eval_dict

    def run_n_episodes(self, n, iteration):
        if self.algorithm == "ACDPI":
            eval_list = [self.run_an_episode(iteration, self.render) for _ in range(n)]
            avg_idsim_tb_eval_dict = {
                k: np.mean([d[k] for d in eval_list]) for k in eval_list[0].keys()
                }
            self.new_lambda = np.mean(self.lambda_list)
            self.mean_mul = np.mean(self.mean_mul_list)
            return avg_idsim_tb_eval_dict, self.new_lambda , self.mean_mul
        else:
           eval_list = [self.run_an_episode(iteration, self.render) for _ in range(n)]
           avg_idsim_tb_eval_dict = {
               k: np.mean([d[k] for d in eval_list]) for k in eval_list[0].keys()
               }
           return avg_idsim_tb_eval_dict

    
    def save_eval_scenario(self, eval_result: EvalResult):
        selected, done_info = self.filter_eval_scenario(eval_result)
        if selected:
            with open(self.save_folder  + '/scene_info.json', 'a') as f:
                    # record scene info
                    scenario_info = {
                        "iteration": eval_result.iteration,
                        "scenario_root": str(pathlib.Path(eval_result.map_path).parent),
                        "map_id": eval_result.map_id,
                        "seed": eval_result.seed,
                        "traffic_seed": int(eval_result.traffic_seed),
                        "ego_id": eval_result.ego_id,
                        "warmup_time": eval_result.warmup_time,
                        "done_info": done_info
                    }
                    json.dump(scenario_info, f, indent=4)
                    f.write(',\n')
        else:
            pass
        return

    def filter_eval_scenario(self, eval_result: EvalResult):
        # filter the scenario that we want to save
        collision = eval_result.done_info['done/collision']
        off_road = eval_result.done_info['done/out_of_driving_area']

        selected = collision or off_road
        if selected:
            done_info = 'collision' if collision else 'off_road'
        else:
            done_info = None
        return selected, done_info

    def fluctuation(self, list):
        a_t = np.array(list[1:])
        a_t_last = np.array(list[:-1])
        self.fluctuation_rate = np.linalg.norm(a_t - a_t_last, ord=2, axis=-1).mean()
        return self.fluctuation_rate


        