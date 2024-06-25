#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Evaluator for IDSim when test
#  Update Date: 2023-12-14, Guojian Zhan: create this file

from typing import Dict, List, Tuple, NamedTuple
import json
import numpy as np
import torch
import pickle
import pathlib
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

class IDCConfig(NamedTuple):
    lane_change_cooldown: int = 150
    lane_change_channeling: int = 10

class EvalResult:
    def __init__(self):
        self.map_path: str = None
        self.map_id: str = None
        self.seed: int = None
        self.warmup_time: float = None
        self.save_folder: str = None
        self.ego_id: str = None
        self.ego_route: Tuple = None
        self.time_stamp_list: List[float] = []
        self.ego_state_list: List[np.ndarray] = [] # x, y, vx, vy, phi, r
        self.reference_list: List[np.ndarray] = []
        self.surr_state_list: List[np.ndarray] = []
        self.surrounding_vehicles: List[SurroundingVehicle] = []
        self.context_list: List[BaseContext] = []
        self.context_full_list: List[BaseContext] = []
        self.obs_list: List[np.ndarray] = []
        self.attn_weight_list: List[np.ndarray] = []
        self.action_list: List[np.ndarray] = []
        self.action_real_list: List[np.ndarray] = []
        self.reward_list: List[float] = []
        self.value_list: List[float] = []
        ## IDC
        self.selected_path_index_list: List[int] = []
        self.optimal_path_index_list: List[int] = []
        self.paths_value_list: List[List[float]] = []
        self.ref_allowable: List[List[bool]] = []
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
        kwargs.update({
            "reward_scale": None,
            "repeat_num": 1,
            "act_seq_len": 1,
            "gym2gymnasium": False,
            "vector_env_num": None,
        })
        self.kwargs = kwargs
        # update env_config in kwargs
        env_config = {**self.kwargs['env_config'],
                      'logging_root': self.kwargs['save_folder'], 'scenario_selector': str(0)}
        self.kwargs = {**self.kwargs, 'env_config': env_config}
        super().__init__(index, **self.kwargs)
        self.networks.cpu()  #  for convenience
        # self.env: idSimEnv = create_env(**self.kwargs)
        self.envmodel: idSimEnvModel = create_env_model(**kwargs)
        self.kwargs["action_high_limit"] = self.env.action_space.high
        self.kwargs["action_low_limit"] = self.env.action_space.low

        # eval
        self.IDC_MODE = self.kwargs.get("IDC_MODE", False)
        if self.IDC_MODE:
            self.PATH_SELECTION_EVIDENCE = self.kwargs["PATH_SELECTION_EVIDENCE"]
            self.idc_config = IDCConfig()
            self.PATH_SELECTION_DIFF_THRESHOLD = self.kwargs["PATH_SELECTION_DIFF_THRESHOLD"]
            self.fast_mode = self.kwargs.get("fast_mode", True)
        self.eval_PODAR = self.kwargs.get("eval_PODAR", False)
        self.num_eval_episode = self.kwargs["num_eval_episode"]
        self.eval_save = self.kwargs.get("eval_save", True)
        self.save_folder = self.kwargs["save_folder"]
        self.use_mpc = False

        if kwargs["ini_network_dir"] is not None:
            self.networks.load_state_dict(
                torch.load(self.kwargs["ini_network_dir"]))
    
    def idc_decision(self,
                     idc_env_info: Tuple,
                     last_optimal_path_index: int,
                     selected_path_index: int,
                     episode_step: int,
                     lc_cd: int,
                     lc_cl: int,
                     eval_result: EvalResult):
        cur_index, lane_list = idc_env_info

        paths_value_list = [0.] * len(lane_list)
        ref_allowable = [False] * len(lane_list)
        allowable_ref_index_list = get_allowable_ref_list(cur_index, lane_list)
        if selected_path_index not in allowable_ref_index_list:
            allowable_ref_index_list.append(selected_path_index)
        allowable_ref_value = []
        allowable_context_list = []
        if lc_cd < self.idc_config.lane_change_cooldown and self.fast_mode:
            allowable_ref_index_list = [selected_path_index]
            context = MultiLaneContext.from_env(self.env, self.env.model_config, selected_path_index)
            context = stack_samples([context])
            value = 999
            allowable_ref_value.append(value)
            allowable_context_list.append(context)
        else:
            for ref_index in allowable_ref_index_list:
                value, context = self.eval_ref_by_index(ref_index)
                if ref_index == selected_path_index:
                    value += self.PATH_SELECTION_DIFF_THRESHOLD
                allowable_ref_value.append(value)
                allowable_context_list.append(context)
        # find optimal path: safe and max value, default selected path
        optimal_path_index = selected_path_index
        optimal_path_in_allowable = allowable_ref_index_list.index(optimal_path_index)
        optimal_value = allowable_ref_value[optimal_path_in_allowable]
        for i, ref_index in enumerate(allowable_ref_index_list):
            if allowable_ref_value[i] > optimal_value:
                optimal_path_index = ref_index
                optimal_value = allowable_ref_value[i]

        if optimal_path_index == selected_path_index:
            new_selected_path_index = selected_path_index
            lc_cd += 1
            lc_cl = 0
        else:
            print(f'Lane Changing: {selected_path_index} -> {optimal_path_index}')
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
            if lc_cd < self.idc_config.lane_change_cooldown:
                print(f'    [Shutdown] lc_cd={lc_cd}<{self.idc_config.lane_change_cooldown}')
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
                        print(f'    [Success] {selected_path_index} -> {optimal_path_index}')
                        new_selected_path_index = optimal_path_index
                        lc_cd = 0
                        lc_cl = 0

        for i, ref_index in enumerate(allowable_ref_index_list):
            paths_value_list[ref_index] = allowable_ref_value[i]
            ref_allowable[ref_index] = True

        context = allowable_context_list[allowable_ref_index_list.index(new_selected_path_index)]

        # save
        eval_result.lc_cl.append(lc_cl)
        eval_result.lc_cd.append(lc_cd)
        eval_result.paths_value_list.append(deepcopy(paths_value_list))
        eval_result.ref_allowable.append(deepcopy(ref_allowable))
        eval_result.context_list.append(context)
        if new_selected_path_index != selected_path_index:
            eval_result.lane_change_step.append(episode_step)

        return optimal_path_index, new_selected_path_index, lc_cd, lc_cl

    def get_idsim_context(self, index):
        if self.env.scenario == "crossroad":
            idsim_context = CrossRoadContext.from_env(self.env, self.env.model_config, index)
        elif self.env.scenario == "multilane":
            idsim_context = MultiLaneContext.from_env(self.env, self.env.model_config, index)
        return idsim_context
    
    def get_pyth_idsim_state(self, idsim_context):
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
                boundary_param=idsim_context.p.boundary_param,
                real_t = torch.tensor(idsim_context.t).int(),
                t = torch.tensor(idsim_context.i).int()
            )
        )
        return state

    def eval_ref_by_index(self, index):
        idsim_context = self.get_idsim_context(index)
        state = self.get_pyth_idsim_state(idsim_context)
        idsim_context_batch = stack_samples([idsim_context])
        model_obs = self.envmodel.idsim_model.observe(idsim_context_batch)
        if self.kwargs.get('obs_scale') is not None:  # NOTE: A temporary solution for obs scale, not good
            scaled_obs = self.env.observation(model_obs)
        else:
            scaled_obs = model_obs

        info = {'state': State.stack([state])}
        d = torch.tensor([0.])
        with torch.no_grad():
            if self.PATH_SELECTION_EVIDENCE == "loss":
                v_pi = torch.tensor([0.])
                if self.kwargs['algorithm'] == 'FHADP2':
                    a_full = self.networks.policy.forward_all_policy(model_obs)
                    for step in range(self.envmodel.idsim_model.N):
                        a = a_full[:, step] # [B, 2]
                        model_obs, r, d, info = self.envmodel.forward(model_obs, a, d, info)
                        r_details = info["reward_details"]
                        v_pi += r
                elif self.kwargs['algorithm'] == "DSACT" or self.kwargs['algorithm'] == "DSACTPI" or self.kwargs['algorithm'] == "DSACTPIR":
                    for step in range(self.envmodel.idsim_model.N):
                        logits = self.networks.policy(scaled_obs)
                        action_distribution = self.networks.create_action_distributions(logits)
                        action = action_distribution.mode().float()
                        model_obs, r, d, info = self.envmodel.forward(model_obs, action, d, info)
                        r_details = info["reward_details"]
                        v_pi += r
                else:
                    raise NotImplementedError

                value = v_pi.item()
            elif self.PATH_SELECTION_EVIDENCE == "value":
                if self.kwargs['algorithm'] == "DSACT" or self.kwargs['algorithm'] == "DSACTPI":
                    logits = self.networks.policy(scaled_obs)
                    action_distribution = self.networks.create_action_distributions(logits)
                    action = action_distribution.mode().float()
                    value = torch.min(self.networks.q1(scaled_obs,action)[:,0], self.networks.q2(scaled_obs,action)[:,0]).item() /100
                elif self.kwargs['algorithm'] == "DSACTPIR":
                    logits = self.networks.policy(scaled_obs)
                    action_distribution = self.networks.create_action_distributions(logits)
                    action = action_distribution.mode().float()
                    v_pi = torch.min(self.networks.q1.cal_comp(scaled_obs,action)[0,10], 
                                     self.networks.q2.cal_comp(scaled_obs,action)[0,10]) /100
                    value = v_pi.item()
                else:
                    value = self.networks.v(scaled_obs).item()
            else:
                raise NotImplementedError
        if self.use_mpc:
            action, res = self.opt_controller(state)
            value = - res.fun
        return value, idsim_context_batch

    def run_an_episode(self, render=False, batch=0, episode=0):
        obs, info = self.env.reset()
        warmup_time = self.env.engine.context.simulation_time
        env_context = self.env.engine.context
        vehicle = env_context.vehicle
        eval_result = EvalResult()
        eval_result.map_path = str(env_context.scenario.root)
        eval_result.map_id = str(self.env.config.scenario_selector)
        eval_result.seed = self.env.config.seed
        eval_result.warmup_time = self.env.config.warmup_time
        eval_result.save_folder = str(self.save_folder)
        eval_result.ego_id = str(vehicle.id)
        eval_result.ego_route = vehicle.route
        
        idsim_tb_eval_dict = {key: 0. for key in idsim_tb_tags_dict.keys()}
        
        done = 0
        info["TimeLimit.truncated"] = False

        # if self.IDC_MODE:  # IDC mode may extremely slow down the evaluation
        if self.env.scenario == "multilane":
            try:  # temporary solution 
                lane_list = env_context.scenario.network.get_edge_lanes(
                    vehicle.edge, vehicle.v_class)
                cur_index = lane_list.index(vehicle.lane)
                lc_cd, lc_cl = 0, 0
                last_optimal_path_index = cur_index
                selected_path_index = cur_index
            except:
                cur_index = 0
                lane_list = [0, 1, 2]
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
            # ----------- select path ------------
            if self.IDC_MODE:
                # idc env info
                if self.env.scenario == "multilane":
                    try:  # temporary solution
                        cur_index = lane_list.index(vehicle.lane)
                    except:
                        cur_index = 0
                else:
                    cur_index = selected_path_index
                idc_env_info = (cur_index, lane_list)
                # idc decision
                optimal_path_index, new_selected_path_index, lc_cd, lc_cl = self.idc_decision(
                    idc_env_info, last_optimal_path_index, selected_path_index,
                    episode_step, lc_cd, lc_cl, eval_result
                )
                # update last_optimal_path_index
                last_optimal_path_index = optimal_path_index
                selected_path_index = new_selected_path_index
            else:
                if self.env.scenario == "multilane":
                    try:  # temporary solution
                        selected_path_index = lane_list.index(vehicle.lane)
                    except:
                        selected_path_index = 0
                else:
                    selected_path_index = 0

            # ----------- get obs ------------
            if self.env.scenario == "crossroad":
                idsim_context = CrossRoadContext.from_env(self.env, self.env.model_config, selected_path_index)
            elif self.env.scenario == "multilane":
                idsim_context = MultiLaneContext.from_env(self.env, self.env.model_config, selected_path_index)
            idsim_context_batch = stack_samples([idsim_context])
            raw_obs = self.env.model.observe(idsim_context_batch)

            if self.kwargs.get('obs_scale') is not None:  # NOTE: A temporary solution for obs sscale, not good
                scaled_obs = self.env.observation(raw_obs)
            else:
                scaled_obs = raw_obs

            # ----------- get action ------------
            logits = self.networks.policy(scaled_obs)
            action_distribution = self.networks.create_action_distributions(logits)
            action = action_distribution.mode()
            # entropy = action_distribution.entropy()
            # print(f"entropy: {entropy.item()}")
            if self.kwargs["algorithm"].startswith("DSACT"):
                q1_value_std = self.networks.q1(scaled_obs, action.float())
                q2_value_std = self.networks.q2(scaled_obs, action.float())
                q1_value = q1_value_std[:, 0]
                q2_value = q2_value_std[:, 0]
                q1_std = q1_value_std[:, 1]
                q2_std = q2_value_std[:, 1]
                q_value = min(q1_value, q2_value).item()
                # print(f"q1_std: {q1_std.item()} q2_std: {q2_std.item()}")

            else:
                q_value = -999  # TODO: fix this
            action = action.detach().numpy()[0]

            if hasattr(self.networks.policy, "pi_net"):
                if hasattr(self.networks.policy.pi_net, "attn_weights") and self.networks.policy.pi_net.attn_weights is not None:
                    attn_weight = self.networks.policy.pi_net.attn_weights
                    eval_result.attn_weight_list.append(attn_weight.detach().numpy()[0])

            # ----------- use mpc to get action ------------
            if self.use_mpc:
                state = self.get_pyth_idsim_state(idsim_context)
                action, res = self.opt_controller(state)

            # ----------- step ------------
            self.env.set_ref_index(selected_path_index)
            next_obs, reward, done, info = self.env.step(action)

            # ----------- save to list------------
            eval_result.obs_list.append(raw_obs)
            eval_result.action_list.append(action)
            eval_result.action_real_list.append(info['state'].robot_state[..., -2:])
            eval_result.value_list.append(q_value)

            eval_result.ego_state_list.append(
            idsim_context.x.ego_state.numpy())
            eval_result.reference_list.append(
            idsim_context.p.ref_param.numpy())
            eval_result.surr_state_list.append(
            idsim_context.p.sur_param.numpy())
            eval_result.time_stamp_list.append(idsim_context.t)
            eval_result.selected_path_index_list.append(selected_path_index)
            for k in eval_result.reward_info.keys():
                if k in info.keys() and  ((k.startswith("env_scaled") or k in ["env_speed_error","env_tracking_error","env_delta_phi"]) or k.startswith("state")):
                    eval_result.reward_info[k].append(info[k])
            obs = next_obs

            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            for eval_key in idsim_tb_eval_dict.keys():
                if eval_key in info.keys() and ((eval_key.startswith("env_scaled") or eval_key in ["env_speed_error","env_tracking_error","env_delta_phi"]) or eval_key.startswith("state")):
                    idsim_tb_eval_dict[eval_key] += info[eval_key]
            # Draw environment animation
            if render:
                self.env.render()
            eval_result.reward_list.append(reward)
            episode_step += 1
        episode_return = sum(eval_result.reward_list)
        idsim_tb_eval_dict["total_avg_return"] = episode_return
        for k, v in idsim_tb_eval_dict.items():
            if k.startswith("done"):
                eval_result.done_info[k] = v
        if self.eval_save:
            with open(self.save_folder + "/{}/episode{}".format('%03d' % batch, '%03d' % episode) + '_eval_dict.pkl', 'wb') as f:
                pickle.dump(eval_result, f, -1)
            with open(self.save_folder + "/{}/episode{}".format('%03d' % batch, '%03d' % episode) + 'scene_info.json', 'w') as f:
                # record scene info
                # record the parent dir of mappath
                scenario_info = {
                    "scenario_root": str(pathlib.Path(eval_result.map_path).parent),
                    "map_id": self.env.config.scenario_selector,
                    "seed": self.env.config.seed,
                    "ego_id": eval_result.ego_id,
                    "warmup_time": warmup_time,
                    "traffic_seed": int(self.env.engine.context.traffic_seed),
                }
                json.dump(scenario_info, f, indent=4)


        return idsim_tb_eval_dict

    def run_n_episodes(self, n):
        batch = 0
        eval_list = []
        for episode in range(n):
            print("##### episode {} #####".format(episode)) 
            idsim_tb_eval_dict = self.run_an_episode(self.render, batch, episode)
            if (episode>0) and ((episode+1) % self.kwargs['env_config']['scenario_reuse'] == 0) \
                or self.kwargs['env_config']['scenario_reuse'] == 1:
                batch += 1
                batch = batch % self.kwargs['env_config']['num_scenarios']
                env_config = {
                    **self.kwargs['env_config'], 'logging_root': self.kwargs['save_folder'], 'scenario_selector': str(batch)}
                kwargs = {**self.kwargs, 'env_config': env_config, "vector_env_num": None,"gym2gymnasium":False}
                self.env.close()
                self.env = create_env(**kwargs)
            eval_list.append(idsim_tb_eval_dict)
        avg_idsim_tb_eval_dict = {
            k: np.mean(np.array([d[k] for d in eval_list])) for k in idsim_tb_eval_dict.keys()
            }
        for k, v in avg_idsim_tb_eval_dict.items():
            print(k, v)
            print('/n')
        return avg_idsim_tb_eval_dict
    
    def filter_episode(self, eval_result: EvalResult):
        return eval_result.done_info['done/arrival']
    
    def save_testcase(self, eval_folder: str, enable_filter_condition: bool = False):
        import os
        batch_folder_list = [i for i in os.listdir(eval_folder) if os.path.isdir(os.path.join(eval_folder, i))]
        test_case_list = []
        for batch_folder in batch_folder_list:
            epi_path = os.path.join(eval_folder, batch_folder) + f'/episode{batch_folder}_eval_dict.pkl'
            with open(epi_path, 'rb') as f:
                eval_result = pickle.load(f)
            if enable_filter_condition and self.filter_episode(eval_result):
                continue
            else:
                test_case = {
                    'scenario_root': eval_result.map_path,
                    'map_id': eval_result.map_id,
                    'seed': eval_result.seed,
                    'warmup_time': eval_result.warmup_time,
                    'ego_id': eval_result.ego_id,
                    'done_info': {k: v for k, v in eval_result.done_info.items() if v > 0},
                }
                test_case_list.append(test_case)
        with open(eval_folder + '/test_case.json', 'w') as f:
            json.dump(test_case_list, f, indent=4)
    
    def run_testcase(self, idx: int, test_case: Dict, use_mpc: bool = False):
        scenario_root = pathlib.Path(test_case['scenario_root'])
        self.save_folder = self.kwargs['save_folder'] + '/test_' + str(idx)
        env_config = {
                    **self.kwargs['env_config'], 
                    'logging_root': self.save_folder, 
                    'scenario_root' : scenario_root,
                    'scenario_selector' : None,
                    'seed' : test_case['seed'],
                    'warmup_time' : test_case['warmup_time'],
                    'ego_id' : test_case['ego_id'],
                    'num_scenarios' :  self.kwargs['env_config']['num_scenarios'],
                    'scenario_reuse' :  1,
                }
        kwargs = {**self.kwargs, 'env_config': env_config}
        self.env.close()
        self.env = create_env(**kwargs)
        self.envmodel: idSimEnvModel = create_env_model(**kwargs)
        if use_mpc:
            from gops.sys_simulator.opt_controller_for_gen_env import OptController
            opt_args={
                "num_pred_step": self.envmodel.idsim_model.N,
                "gamma": 1,
                "mode": "shooting",
                "minimize_options": {"max_iter": 200, "tol": 1e-3,
                                    "acceptable_tol": 1e0,
                                    "acceptable_iter": 10,},
                "use_terminal_cost": False,
            }
            self.opt_controller = OptController(self.envmodel, **opt_args)
            self.opt_controller.return_res = True
            self.use_mpc = True
        idsim_tb_eval_dict = self.run_an_episode(self.render, batch=int(test_case['map_id']), episode=0)
        return idsim_tb_eval_dict