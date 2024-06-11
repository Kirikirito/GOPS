from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Generic, Optional, Tuple, Union
from typing_extensions import Self

import gym
import time
import numpy as np
import torch
from gops.env.env_gen_ocp.pyth_base import (Context, ContextState, Env, State, stateType)
from gops.env.env_gen_ocp.resources.idsim_tags import reward_tags
from idsim.config import Config
from idsim.envs.env import CrossRoad
from idsim_model.model import IdSimModel
from idsim_model.model_context import Parameter, BaseContext
from idsim_model.crossroad.context import CrossRoadContext
from idsim_model.multilane.context import MultiLaneContext
from idsim_model.model_context import State as ModelState
from idsim_model.params import ModelConfig

def cal_ave_exec_time(print_interval=100):
    def decorator(func):
        total_time = 0
        count = 0

        def wrapper(*args, **kwargs):
            nonlocal total_time, count
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            total_time += execution_time
            count += 1
            if count % print_interval == 0:
                print(f"Average execution time after {count} steps: {total_time / count:.9f} seconds")
            return result
        return wrapper
    return decorator

@dataclass
class idSimContextState(ContextState[stateType], Generic[stateType]):
    light_param: Optional[stateType] = None
    ref_index_param: Optional[stateType] = None
    boundary_param: Optional[stateType] = None
    real_t: Union[int, stateType] = 0


class idSimContext(Context):
    def reset(self) -> idSimContextState[np.ndarray]:
        pass

    def step(self) -> idSimContextState[np.ndarray]:
        pass

    def get_zero_state(self) -> idSimContextState[np.ndarray]:
        pass
    

class idSimEnv(CrossRoad, Env):
    def __new__(cls, env_config: Config, model_config: Dict[str, Any], 
                scenario: str, rou_config: Dict[str, Any]=None, env_idx: int=None, scenerios_list: List[str]=None):
        return super(idSimEnv, cls).__new__(cls, env_config)
    
    def __init__(self, env_config: Config, model_config: ModelConfig, 
                 scenario: str, rou_config: Dict[str, Any]=None,env_idx: int=None, scenerios_list: List[str]=None):
        self.env_idx = env_idx  
        print('env_idx:', env_idx)
        self.rou_config = rou_config
        self.env_config = env_config
        self.rou_config = rou_config
        self.change_scenarios(env_idx, scenerios_list)
        super(idSimEnv, self).__init__(env_config)
        self.model_config = model_config
        model_config = deepcopy(model_config)
        self.scenario = scenario

        self._state = None
        self._info = {"reward_comps": np.zeros(len(model_config.reward_comps), dtype=np.float32)}
        self._reward_comp_list = model_config.reward_comps
        # get observation_space
        self.model = IdSimModel(env_config, model_config)
        obs_dim = self.model.obs_dim
        self.use_random_ref_param = env_config.use_multiple_path_for_multilane
        self.random_ref_probability = env_config.random_ref_probability
        self.random_ref_v = env_config.random_ref_v
        self.ref_v_range = env_config.ref_v_range

        if self.use_random_ref_param > 0.0:
            print(f'INFO: randomly choosing reference when resetting env')
        if self.random_ref_probability > 0.0:
            print(f'INFO: randomly choosing reference when stepping at P={self.random_ref_probability}')
        if env_config.takeover_bias:
            print('INFO: using takeover bias True')
        if env_config.use_random_acc:
            print('INFO: using random acceleration')
        if model_config.track_closest_ref_point:
            print('INFO: tracking closest reference point')

        self.lc_cooldown = self.env_config.random_ref_cooldown
        self.lc_cooldown_counter = 0

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.context = idSimContext() # fake idsim context
        self.set_scenario(scenario)
        self.ref_index = None

    def seed(self, seed=None):
        super(idSimEnv, self).seed(seed)
        
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        self.lc_cooldown_counter = 0
        if self.rou_config is not None:
            if self.engine is None:
                # first create engine
                print("INFO: change rou")
                self.change_rou_file()
            else:
                print(self.engine.context.episode_count, self.config.scenario_reuse)
                if (self.engine.context.episode_count+1) % self.config.scenario_reuse == 0:
                    # need to change new map
                    print("INFO: change rou")
                    self.change_rou_file()
        obs, info = super(idSimEnv, self).reset(**kwargs)
        env_context = self.engine.context
        vehicle = env_context.vehicle
        self.ref_index = None
        if self.scenario == "multilane" and self.use_random_ref_param and not env_context.vehicle.in_junction:
            lane_list = env_context.scenario.network.get_edge_lanes(
                vehicle.edge, vehicle.v_class)
            cur_index = lane_list.index(vehicle.lane)
            self.ref_index = np.random.choice([0,1,-1]) + cur_index # only allow lane change by 1
            self.ref_index = np.clip(self.ref_index, 0, len(lane_list)-1)
        else:
            self.ref_index = np.random.choice(
                np.arange(self.model_config.num_ref_lines)
            ) if self.use_random_ref_param else None
        if self.random_ref_v and not env_context.vehicle.in_junction:  # TODO: check if this is correct
            ref_v = np.random.uniform(*self.ref_v_range)
            self.model_config.ref_v_lane = float(ref_v)
            self.env_config.ref_v = float(ref_v)
            # print(f"INFO: change ref_v to {ref_v}")
            
        self._state = self._get_state_from_idsim(ref_index_param=self.ref_index)
        self._fix_state()
        self._info = self._get_info(info)
        return self._get_obs(), self._info
    
    def _fix_state(self,):  # FIXME: this is a hack
        context = get_idsimcontext(
            State.stack([self._state]), mode="batch", scenario=self.scenario)
        ego_state = context.x.ego_state[0] # [6]: x, y, vx, vy, phi, r
        ref_param = context.p.ref_param[0] # [R, 2N+1, 4] ref_x, ref_y, ref_phi, ref_v
        ref_index = context.p.ref_index_param[0]
        nominal_steer = self._get_nominal_steer_by_state(
            ego_state, ref_param, ref_index)
        if self.engine.context.vehicle.in_junction and np.abs(nominal_steer) > 5*np.pi/180:
            # steer in gaussian distribution
            random_steer = np.random.normal(loc=0.0, scale=3*np.pi/180)
            init_steer = 0.3*nominal_steer+ np.sign(nominal_steer)*np.abs(random_steer)
            steer_upper_bound = self.config.real_action_upper_bound[1]
            steer_lower_bound = self.config.real_action_lower_bound[1]
            init_steer = np.clip(init_steer, steer_lower_bound, steer_upper_bound)
            
            init_acc = np.random.normal(loc=0.0, scale=0.1)
            init_action = np.array([init_acc, init_steer], dtype=np.float32)
            self.engine.context.vehicle.init_act(init_action)

            init_vx = np.random.uniform(0.5, 2.0)
            self.engine.context.vehicle.init_vx(init_vx)
            print(f"INFO: fix state, init_action: {init_action}")
            self._state = self._get_state_from_idsim(ref_index_param=self.ref_index)
    # @cal_ave_exec_time(print_interval=1000)
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminated, truncated, info = super(idSimEnv, self).step(action)

        # ----- cal the next_obs, reward -----
        self.lc_cooldown_counter += 1
        if self.lc_cooldown_counter > self.lc_cooldown:
            # lane change is allowable
            if not self.engine.context.vehicle.in_junction and self.use_random_ref_param and np.random.rand() < self.random_ref_probability :
                new_ref_index = np.random.choice(np.arange(self.model_config.num_ref_lines))
                if new_ref_index != self.ref_index:
                    # lane change
                    self.ref_index = new_ref_index
                    self.lc_cooldown_counter = 0
        reward_model, reward_details = self._get_reward(action)
        self._state = self._get_state_from_idsim(ref_index_param=self.ref_index)
        reward_model_free, mf_info = self._get_model_free_reward(action)
        info.update(mf_info)

        info["reward_details"] = dict(
            zip(reward_tags, [i.item() for i in reward_details])
        )
        done = terminated or truncated
        if truncated:
            info["TimeLimit.truncated"] = True # for gym

        self._info = self._get_info(info)
        total_reward = reward + reward_model_free
        # if not terminated:
        #     total_reward = np.maximum(total_reward, 0.05)

        return self._get_obs(), total_reward, done, self._info

    def set_ref_index(self, ref_index: int):
        self.ref_index = ref_index
    
    def _get_info(self, info) -> dict:
        info.update(Env._get_info(self))
        if "env_reward_step" in info.keys():
            info["reward_comps"] = np.array([info[i] for i in self._reward_comp_list], dtype=np.float32)
        else:
            info["reward_comps"] = np.zeros(len(self._reward_comp_list), dtype=np.float32)
        return info
    
    @property
    def additional_info(self) -> dict:
        info = super().additional_info
        info.update({
            "reward_comps":{
                "shape":(len(self._reward_comp_list),), 
                "dtype":np.float32
            }
        })
        return {}
    
    def _get_obs(self) -> np.ndarray:
        idsim_context = get_idsimcontext(
            State.stack([self._state.array2tensor()]), mode="batch", scenario=self.scenario)
        model_obs = self.model.observe(idsim_context)
        return model_obs.numpy().squeeze(0)

    def _get_reward(self, action: np.ndarray) -> Tuple[float, Tuple]:
        cur_state = self._state.array2tensor()
        next_state = self._get_state_from_idsim(ref_index_param=self.ref_index).array2tensor()
        idsim_context = get_idsimcontext(State.stack([next_state]), mode="batch", scenario=self.scenario)
        action = torch.tensor(action)
        reward_details = self.model.reward_nn_state(
            context=idsim_context,
            last_last_action=cur_state.robot_state[..., -4:-2].unsqueeze(0), # absolute action
            last_action=cur_state.robot_state[..., -2:].unsqueeze(0), # absolute action
            action=action.unsqueeze(0) # incremental action
        )
        return reward_details[0].item(), reward_details
    
    def _get_model_free_reward(self, action: np.ndarray) -> float:
        idsim_context = get_idsimcontext(
            State.stack([self._state]), 
            mode="batch", 
            scenario=self.scenario
        )
        reward, info = self.model_free_reward(
            context=idsim_context,
            last_last_action=self._state.robot_state[..., -4:-2][None, :], # absolute action
            last_action=self._state.robot_state[..., -2:][None, :], # absolute action
            action=action[None, :] # incremental action
        )
        return reward, info
    
    def _get_terminated(self) -> bool:
        """abandon this function, use terminated from idsim instead"""
        ...
    
    def _get_state_from_idsim(self, ref_index_param=None) -> State:
        if self.scenario == "crossroad":
            idsim_context = CrossRoadContext.from_env(self, self.model_config, ref_index_param)
        elif self.scenario == "multilane":
            idsim_context = MultiLaneContext.from_env(self, self.model_config, ref_index_param)
        else:
            raise NotImplementedError

        return State(
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
        ).tensor2array()

    def get_zero_state(self) -> State[np.ndarray]:
        if self._state is None:
            self.reset()
        return State(
            robot_state=np.zeros_like(self._state.robot_state, dtype=np.float32),
            context_state=idSimContextState(
                reference=np.zeros_like(self._state.context_state.reference, dtype=np.float32),
                constraint=np.zeros_like(self._state.context_state.constraint, dtype=np.float32),
                t=np.zeros_like(self._state.context_state.t, dtype=np.int64),
                light_param=np.zeros_like(self._state.context_state.light_param, dtype=np.float32),
                ref_index_param=np.zeros_like(self._state.context_state.ref_index_param, dtype=np.int64),
                boundary_param=np.zeros_like(self._state.context_state.boundary_param, dtype=np.float32),
                real_t=np.zeros_like(self._state.context_state.real_t, dtype=np.int64)
            )
        )
    # close
    def close(self) -> None:
        super(idSimEnv, self).close()

    def change_scenarios(self, idx: int, scenario_list: List[str]) -> None:
        if idx is None  or  scenario_list is None: # TODO: more elegant way to handle this
            print(f"INFO: no change in scenario")
            return
        scenarios = scenario_list[idx% len(scenario_list)]
        self.env_config.scenario_selector = scenarios
        print(f"INFO: change current scenario to {scenarios}")
        # direction_list = [ "r", "l", None]
        # self.env_config.direction_selector = direction_list[(idx // len(scenario_list))%len(direction_list)] # FIXME: this is a hack
        # print(f"INFO: change current direction to {self.env_config.direction_selector}")
        # direction_list = ["r", "s"]
        # self.env_config.direction_selector = direction_list[idx % len(direction_list)] # FIXME: this is a hack
        # print(f"INFO: change current direction to {self.env_config.direction_selector}")

    def change_rou_file(self):
        surrounding_max_speed_range = self.rou_config["surrounding_max_speed_range"]
        surrounding_max_speed_list = np.random.uniform(*surrounding_max_speed_range, size=5)
        # surrounding_max_speed_list = [4.0, 5.0, 6.0]
        print(f"INFO: change surrounding_max_speed to {surrounding_max_speed_list}")
        change_dict = {"maxSpeed": surrounding_max_speed_list}
        if self.env_config.scenario_selector is not None:
            map_path = self.env_config.scenario_root / self.env_config.scenario_selector
            map_path_list = [map_path]
        else:
            map_path_list = [map_path for map_path in self.env_config.scenario_root.iterdir() if map_path.is_dir()]
        for map_path in map_path_list:
            rou_path = map_path / "scene.rou.xml"
            assert rou_path.exists(), f"rou_path {rou_path} does not exist"
            change_rou(rou_path, change_dict)

def get_idsimcontext(state: State, mode: str, scenario: str) -> BaseContext:
    if scenario == "crossroad":
        Context = CrossRoadContext
    elif scenario == "multilane":
        Context = MultiLaneContext
    else:
        raise NotImplementedError
    if mode == "full_horizon":
        context = Context(
            x = ModelState(
                ego_state = state.robot_state[..., :-4],
                last_last_action = state.robot_state[..., -4:-2],
                last_action = state.robot_state[..., -2:]
            ),
            p = Parameter(
                ref_param = state.context_state.reference,
                sur_param = state.context_state.constraint,
                light_param = state.context_state.light_param,
                ref_index_param = state.context_state.ref_index_param,
                boundary_param=state.context_state.boundary_param
            ),
            t = state.context_state.real_t,
            i = state.context_state.t[0]
        )
    elif mode == "batch":
        if isinstance(state.context_state.t, np.ndarray):
            assert np.unique(state.context_state.t).shape[0] == 1, "batch mode only support same t"
        elif isinstance(state.context_state.t, torch.Tensor):
            assert state.context_state.t.unique().shape[0] == 1, "batch mode only support same t"
        else:
            raise NotImplementedError
        context = Context(
            x = ModelState(
                ego_state = state.robot_state[..., :-4],
                last_last_action = state.robot_state[..., -4:-2],
                last_action = state.robot_state[..., -2:]
            ),
            p = Parameter(
                ref_param = state.context_state.reference,
                sur_param = state.context_state.constraint,
                light_param = state.context_state.light_param,
                ref_index_param = state.context_state.ref_index_param,
                boundary_param=state.context_state.boundary_param
            ),
            t = state.context_state.real_t,
            i = state.context_state.t[0]
        )
    else:
        raise NotImplementedError
    return context

def change_rou(rou_path: Path, change_dict: Dict[str, List[float]]) -> None:
    """
    change surrounding_max_speed in rou_path
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(rou_path)
    root = tree.getroot()
    ind = -1
    for child in root:
        if child.tag == 'vType':
            ind += 1
            print(child.tag, child.attrib)
            # change the attribute
            for k, v in change_dict.items():
                mod_int = ind % len(v)
                child.attrib[k] = str(v[mod_int])
    # write to file
    tree.write(rou_path)

def env_creator(**kwargs):
    """
    make env `pyth_idsim`
    """
    assert "env_config" in kwargs.keys(), "env_config must be specified"
    env_config = deepcopy(kwargs["env_config"])

    assert "env_scenario" in kwargs.keys(), "env_scenario must be specified"
    env_scenario = kwargs["env_scenario"]

    assert 'scenario_root' in env_config, "scenario_root must be specified in env_config"
    env_config['scenario_root'] = Path(env_config['scenario_root'])
    env_config = Config.from_partial_dict(env_config)

    assert "env_model_config" in kwargs.keys(), "env_model_config must be specified"
    model_config = deepcopy(kwargs["env_model_config"])
    model_config = ModelConfig.from_partial_dict(model_config)

    rou_config = kwargs["rou_config"] if "rou_config" in kwargs.keys() else None

    env_idx = kwargs["env_idx"] if "env_idx" in kwargs.keys() else 0

    scenerios_list = kwargs["scenerios_list"] if "scenerios_list" in kwargs.keys() else None
    env = idSimEnv(env_config, model_config, env_scenario, rou_config, env_idx, scenerios_list)
    return env