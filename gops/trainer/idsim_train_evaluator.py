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

    def run_an_episode(self, iteration, episode, render=True):
        if self.print_iteration != iteration:
            self.print_iteration = iteration
            self.print_time = 0
        else:
            self.print_time += 1


        idsim_tb_eval_dict = {key: 0. for key in idsim_tb_tags_dict.keys()}
        if self.fixed_eval_seed:
            env_seed = self.seed_list[episode]
        else:
            env_seed = self.env_seed_rng.integers(1, 2**32)
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

        done = 0
        info["TimeLimit.truncated"] = False
        self.obs_buffer.clear()
        while not (done or info["TimeLimit.truncated"]):
            batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
            self.obs_buffer.add(batch_obs)
            with torch.no_grad():
                logits = self.networks.policy(self.obs_buffer.get().to(self.device))
                action_distribution = self.networks.create_action_distributions(logits)
            action = action_distribution.mode()
            action = action.cpu().detach().numpy()[0]
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

        return idsim_tb_eval_dict

    def run_n_episodes(self, n, iteration):
        eval_list = [self.run_an_episode(iteration, i, render=self.render) for i in range(n)]
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
                        "scene": done_info
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
        

        