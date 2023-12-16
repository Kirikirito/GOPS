#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Evaluator for IDSim when test
#  Update Date: 2023-12-14, Guojian Zhan: create this file

from typing import Dict, List, Tuple
import json
import numpy as np
import torch
import pickle
import os
from copy import deepcopy
from gops.trainer.evaluator import Evaluator
from gops.env.env_gen_ocp.resources.idsim_tags import idsim_tb_tags_dict, reward_tags
from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_env_model import create_env_model

from gops.env.env_gen_ocp.pyth_base import (Context, ContextState, Env, State, stateType)
from gops.env.env_gen_ocp.pyth_idsim import idSimEnv, get_idsimcontext, idSimContextState
from gops.env.env_gen_ocp.env_model.pyth_idsim_model import idSimEnvModel


from idsim_model.model_context import BaseContext
from idsim_model.crossroad.context import CrossRoadContext
from idsim_model.multilane.context import MultiLaneContext
from idsim_model.model import IdSimModel
from idsim_model.params import ModelConfig
from idsim_model.utils.model_utils import stack_samples
from idsim.component.vehicle.surrounding import SurroundingVehicle


def get_args_from_json(json_file_path, args_dict):
    summary_filename = json_file_path
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)
    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]
    return args_dict

def get_allowable_ref_list(cur_index, lane_list):
    if len(lane_list) == 1:
        return [cur_index]
    else:
        if cur_index == 0:
            return [cur_index, cur_index + 1]
        elif cur_index == len(lane_list) - 1:
            return [cur_index - 1, cur_index]
        else:
            return [cur_index - 1, cur_index, cur_index + 1]

class EvalResult:
    def __init__(self):
        self.map_path: str = None
        self.save_folder: str = None
        self.ego_id: str = None
        self.ego_route: Tuple = None
        self.time_stamp_list: List[float] = []
        self.ego_state_list: List[np.ndarray] = []
        self.reference_list: List[np.ndarray] = []
        self.surr_state_list: List[np.ndarray] = []
        self.surrounding_vehicles: List[SurroundingVehicle] = []
        self.context_list: List[BaseContext] = []
        self.context_full_list: List[BaseContext] = []
        self.obs_list: List[np.ndarray] = []
        self.action_list: List[np.ndarray] = []
        self.action_real_list: List[np.ndarray] = []
        self.reward_list: List[float] = []
        ## IDC
        self.selected_path_index_list: List[int] = []
        self.paths_value_list: List[List[float]] = []
        self.ref_allowable: List[List[float]] = []
        self.lane_change_step: List[int] = []
        self.lc_cd: List[int] = []
        self.lc_cl: List[int] = []
        ## done
        self.done_info: Dict[str, int] = {}
        ## rewards
        self.reward_info: Dict[str, List[float]] = {k: [] for k in reward_tags}

class IdsimIDCEvaluator(Evaluator):
    def __init__(self, index=0, **kwargs):
        kwargs['env_config']['singleton_mode'] = 'invalidate'
        self.kwargs = kwargs
        # update env_config in kwargs
        env_config = {**self.kwargs['env_config'],
                      'logging_root': self.kwargs['save_folder'], 'scenario_selector': str(0)}
        self.kwargs = {**self.kwargs, 'env_config': env_config}
        super().__init__(index, **self.kwargs)
        # self.env: idSimEnv = create_env(**self.kwargs)
        self.envmodel: idSimEnvModel = create_env_model(**kwargs)
        self.kwargs["action_high_limit"] = self.env.action_space.high
        self.kwargs["action_low_limit"] = self.env.action_space.low

        # eval
        self.IDC_MODE = self.kwargs.get("IDC_MODE", False)
        if self.IDC_MODE:
            self.PATH_SELECTION_EVIDENCE = self.kwargs["PATH_SELECTION_EVIDENCE"]
        self.eval_PODAR = self.kwargs.get("eval_PODAR", False)
        self.num_eval_episode = self.kwargs["num_eval_episode"]
        self.eval_save = self.kwargs.get("eval_save", True)
        self.save_folder = self.kwargs["save_folder"]

        if kwargs["ini_network_dir"] is not None:
            self.networks.load_state_dict(
                torch.load(self.kwargs["ini_network_dir"]))
    
    def idc_decision(self,
                     idc_env_info: Tuple[int, List[str], List[List[float]], List[List[float]]],
                     last_optimal_path_index: int,
                     selected_path_index: int,
                     episode_step: int,
                     lc_cd: int,
                     lc_cl: int,
                     eval_result: EvalResult):
        cur_index, lane_list = idc_env_info

        paths_value_list = [0] * len(lane_list)
        ref_allowable = [False] * len(lane_list)
        allowable_ref_index_list = get_allowable_ref_list(cur_index, lane_list)
        if selected_path_index not in allowable_ref_index_list:
            allowable_ref_index_list.append(selected_path_index)
        allowable_ref_value = []
        allowable_ref_safe = []
        # cal value and safe for allowable ref
        allowable_context_list = []
        allowable_obs_list = []
        allowable_action_list = []
        for ref_index in allowable_ref_index_list:
            value, context, obs, action, reward_tuple = self.eval_ref_by_index(
                ref_index)
            if ref_index == selected_path_index:
                value += 100.
            collision_flag = reward_tuple[-1]
            collision = collision_flag.sum().item() > 0
            allowable_ref_value.append(value)
            allowable_ref_safe.append(not collision)
            allowable_context_list.append(context)
            allowable_obs_list.append(obs)
            allowable_action_list.append(action[:2])
        # find optimal path: safe and max value, default selected path
        optimal_path_index = selected_path_index
        optimal_path_in_allowable = allowable_ref_index_list.index(optimal_path_index)
        optimal_value = allowable_ref_value[optimal_path_in_allowable]
        for i, ref_index in enumerate(allowable_ref_index_list):
            if allowable_ref_safe[i] and allowable_ref_value[i] > optimal_value:
                optimal_path_index = ref_index
                optimal_value = allowable_ref_value[i]

        if optimal_path_index == selected_path_index:
            new_selected_path_index = selected_path_index
            lc_cd += 1
            lc_cl = 0
        else:
            if selected_path_index not in allowable_ref_index_list:
                print("selected path not in allowable ref")
                print("selected_path_index", selected_path_index)
                print("allowable_ref_index_list", allowable_ref_index_list)
                print("lc_cd", eval_result.lc_cd)
                print("lc_cl", eval_result.lc_cl)
                print("episode_step", episode_step)
                print("ego_state_full", eval_result.ego_state_list)
                print("ego_state", self.env.engine.context.vehicle.state)
                print(eval_result.selected_path_index_list)
            selected_path_in_allowable = allowable_ref_index_list.index(
                selected_path_index)
            selected_ref_safe = allowable_ref_safe[selected_path_in_allowable]
            if not selected_ref_safe:
                # emergency
                new_selected_path_index = optimal_path_index
                lc_cd = 0
                lc_cl = 0
            else:
                if lc_cd < self.idc_config.lane_change_cooldown:
                    new_selected_path_index = selected_path_index
                    lc_cd += 1
                    lc_cl = 0
                else:
                    if not optimal_path_index == last_optimal_path_index:
                        new_selected_path_index = selected_path_index
                        lc_cd += 1
                        lc_cl = 0
                    else:
                        if lc_cl < self.idc_config.lane_change_channeling:
                            new_selected_path_index = selected_path_index
                            lc_cd += 1
                            lc_cl += 1
                        else:
                            new_selected_path_index = optimal_path_index
                            lc_cd = 0
                            lc_cl = 0

        for i, ref_index in enumerate(allowable_ref_index_list):
            paths_value_list[ref_index] = allowable_ref_value[i]
            ref_allowable[ref_index] = True

        context = allowable_context_list[allowable_ref_index_list.index(
            new_selected_path_index)]
        obs = allowable_obs_list[allowable_ref_index_list.index(
            new_selected_path_index)]
        action = allowable_action_list[allowable_ref_index_list.index(
            new_selected_path_index)]

        # save
        eval_result.lc_cl.append(lc_cl)
        eval_result.lc_cd.append(lc_cd)
        eval_result.paths_value_list.append(deepcopy(paths_value_list))
        eval_result.ref_allowable.append(deepcopy(ref_allowable))
        eval_result.context_list.append(context)
        if new_selected_path_index != selected_path_index:
            eval_result.lane_change_step.append(episode_step)

        return optimal_path_index, new_selected_path_index, lc_cd, lc_cl, \
            context, obs, action

    def eval_ref_by_index(self, index):
        if self.env.scenario == "crossroad":
            idsim_context = CrossRoadContext.from_env(self.env, self.env.model_config, index)
        elif self.env.scenario == "multilane":
            idsim_context = MultiLaneContext.from_env(self.env, self.env.model_config, index)
        state = State(
            robot_state=torch.concat([
                idsim_context.x.ego_state, 
                idsim_context.x.last_last_action, 
                idsim_context.x.last_action],
            dim=-1),
            context_state=idSimContextState(
                reference=idsim_context.p.ref_param, 
                constraint=idsim_context.p.sur_param,
                light_param=idsim_context.p.light_param, 
                ref_index_param=idsim_context.p.ref_index_param,
                real_t = torch.tensor(idsim_context.t).int(),
                t = torch.tensor(idsim_context.i).int()
            )
        )
        idsim_context = stack_samples([idsim_context])
        model_obs = self.env.model.observe(idsim_context)
        with torch.no_grad():
            if self.PATH_SELECTION_EVIDENCE == "loss":
                action = self.networks.policy(model_obs)[0]
                next_state = self.envmodel.get_next_state(state, action)
                rewards = self.envmodel.idsim_model.reward_nn_state(
                    context=get_idsimcontext(State.stack([next_state]), mode="batch", scenario=self.env.scenario),
                    last_last_action=next_state.robot_state[..., -4:-2].unsqueeze(0), # absolute action
                    last_action=next_state.robot_state[..., -2:].unsqueeze(0), # absolute action
                    action=action.unsqueeze(0) # incremental action
                )
                value = rewards[0].item()
            else:
                rewards = None
                value = self.networks.value(model_obs).item()
        return value, idsim_context, model_obs, action, rewards

    def run_an_episode(self, iteration, render=False, batch=0, episode=0):
        if self.print_iteration != iteration:
            self.print_iteration = iteration
            self.print_time = 0
        else:
            self.print_time += 1
        obs, info = self.env.reset()
        env_context = self.env.engine.context
        vehicle = env_context.vehicle
        eval_result = EvalResult()
        eval_result.map_path = str(env_context.scenario.root)
        eval_result.save_folder = str(self.save_folder)
        eval_result.ego_id = str(vehicle.id)
        eval_result.ego_route = vehicle.route
        
        idsim_tb_eval_dict = {key: 0. for key in idsim_tb_tags_dict.keys()}
        
        done = 0
        info["TimeLimit.truncated"] = False

        # if self.IDC_MODE:  # IDC mode may extremely slow down the evaluation
        if self.env.scenario == "multilane":
            lane_list = env_context.scenario.network.get_edge_lanes(
                vehicle.edge, vehicle.v_class)
            cur_index = lane_list.index(vehicle.lane)
            reference_list = [env_context.scenario.network.get_lane_center_line(
                lane) for lane in lane_list]
            reference_info_list = vehicle.reference_info_list * 3
            lc_cd, lc_cl = 0, 0
            last_optimal_path_index = cur_index
            selected_path_index = cur_index
        else:
            cur_index = 0
            lane_list = [0, 1, 2]
        lc_cd, lc_cl = 0, 0
        last_optimal_path_index = cur_index
        selected_path_index = cur_index

        episode_step = 0
        while not (done or info["TimeLimit.truncated"]):
            if self.IDC_MODE:
                # idc env info
                if self.env.scenario == "multilane":
                    cur_index = lane_list.index(vehicle.lane)
                else:
                    cur_index = selected_path_index
                idc_env_info = (cur_index, lane_list)
                # idc decision
                optimal_path_index, new_selected_path_index, lc_cd, lc_cl, \
                    context, obs, action \
                    = self.idc_decision(
                        idc_env_info,
                        last_optimal_path_index, selected_path_index,
                            episode_step, lc_cd, lc_cl, eval_result)
                # update last_optimal_path_index
                last_optimal_path_index = optimal_path_index
                selected_path_index = new_selected_path_index
            else:
                if self.env.scenario == "multilane":
                    selected_path_index = lane_list.index(vehicle.lane)
                else:
                    selected_path_index = 0

            if self.env.scenario == "crossroad":
                idsim_context = CrossRoadContext.from_env(self.env, self.env.model_config, selected_path_index)
            elif self.env.scenario == "multilane":
                idsim_context = MultiLaneContext.from_env(self.env, self.env.model_config, selected_path_index)
            idsim_context = stack_samples([idsim_context])
            obs = self.env.model.observe(idsim_context)

            logits = self.networks.policy(obs)
            action_distribution = self.networks.create_action_distributions(logits)
            action = action_distribution.mode()
            action = action.detach().numpy()[0]
            next_obs, reward, done, next_info = self.env.step(action)
            eval_result.obs_list.append(obs)
            eval_result.action_list.append(action)

            eval_result.ego_state_list.append(
                idsim_context.x.ego_state.squeeze().numpy())
            eval_result.reference_list.append(
                idsim_context.p.ref_param.squeeze().numpy())
            eval_result.surr_state_list.append(
                idsim_context.p.sur_param.squeeze().numpy())
            eval_result.time_stamp_list.append(idsim_context.t.item())
            eval_result.selected_path_index_list.append(selected_path_index)
            for k, v in eval_result.reward_info.items():
                eval_result.reward_info[k].append(next_info['reward_details'][k])
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
        episode_return = sum(eval_result.reward_list)
        idsim_tb_eval_dict["total_avg_return"] = episode_return
        if self.eval_save:
            with open(self.save_folder + "/{}/episode{}".format('%03d' % batch, '%03d' % episode) + '_eval_dict.pkl', 'wb') as f:
                pickle.dump(eval_result, f, -1)
        return idsim_tb_eval_dict

    def run_n_episodes(self, n, iteration):
        batch = 0
        eval_list = []
        for episode in range(n):
            print("##### episode {} #####".format(episode)) 
            idsim_tb_eval_dict = self.run_an_episode(iteration, self.render, batch, episode)
            if (episode>0) and ((episode+1) % self.kwargs['env_config']['scenario_reuse'] == 0):
                batch += 1
                batch = batch % self.kwargs['env_config']['num_scenarios']
                env_config = {
                    **self.kwargs['env_config'], 'logging_root': self.kwargs['save_folder'], 'scenario_selector': str(batch)}
                kwargs = {**self.kwargs, 'env_config': env_config}
                self.env = create_env(**kwargs)
            eval_list.append(idsim_tb_eval_dict)
        avg_idsim_tb_eval_dict = {
            k: np.mean(np.array([d[k] for d in eval_list])) for k in idsim_tb_eval_dict.keys()
            }
        for k, v in avg_idsim_tb_eval_dict.items():
            print(k, v)
        return avg_idsim_tb_eval_dict