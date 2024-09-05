from typing import Dict, Union, Tuple
import numpy as np

MAP_ROOT_CROSSROAD = '/root/idsim-scenarios/idsim-crossroad'
MAP_ROOT_MULTILANE = '/root/idsim-scenarios/idsim-multilane-cross-dense-v20-mix-multi-size-v3/'
pre_horizon = 30
delta_t = 0.1

env_config_param_base = {
    "use_render": False,
    "seed": 1,
    "actuator": "ExternalActuator",
    "scenario_reuse": 4,
    "num_scenarios": 34,
    "multilane_scenarios": tuple(range(0, 22)),
    "scenario_filter_surrounding_selector": '0,1,2,31,32,33',
    "detect_range": 60,
    "choose_vehicle_retries": 10,
    "choose_vehicle_step_time": 10,
    "scenario_root": MAP_ROOT_CROSSROAD,
    "scenario_selector": None,
     "direction_selector": 's',
    "extra_sumo_args": ("--start", "--delay", "200"),
    "warmup_time": 50.0,
    "max_steps": 200,
    "random_ref_v": True,
    "ref_v_range": (2, 12.0),
    "nonimal_acc": True,
    "ignore_traffic_lights": False,
    "no_done_at_collision": True, 
    "ignore_surrounding": False,
    "ignore_opposite_direction": True,
    "penalize_collision": True,
    "incremental_action": True,
    "action_lower_bound": (-4.0 * delta_t, -0.25 * delta_t),
    "action_upper_bound": (2.5 * delta_t, 0.25 * delta_t),
    "real_action_lower_bound": (-3.0, -0.571),
    "real_action_upper_bound": (0.8, 0.571),
    "obs_num_surrounding_vehicles": {
        "passenger": 8,
        "bicycle": 2,
        "pedestrian": 2,
    },
    "ref_v": 12.0,
    "ref_length": 48.0,
    "obs_num_ref_points": 2 * pre_horizon + 1,
    "obs_ref_interval": 0.8,
    # "vehicle_spec": (1880.0, 1536.7, 1.13, 1.52, -128915.5, -85943.6, 20.0, 0.0),
    "vehicle_spec": (1880.0, 1536.7, 1.22, 1.70, -128915.5, -85943.6, 20.0, 0.0),
    "singleton_mode": "reuse",
    "random_ref_probability": 0.01,
    "use_multiple_path_for_multilane": True,
    "random_ref_cooldown":  80,

    'add_sur_bias': True,
    'sur_bias_range': (1, 2),
    'sur_bias_prob': 0.5,

    "takeover_bias": True, 
    "takeover_bias_prob": 1.0,
    "takeover_bias_x": (0.0, 0.1),
    "takeover_bias_y": (0.0, 0.1),
    "takeover_bias_phi": (0.0, 0.05),
    "takeover_bias_vx": (0.0, 0.0),
    "takeover_bias_ax": (0.0, 0.0),
    "takeover_bias_steer": (0.0, 0.0),
    "minimum_clearance_when_takeover":-1,
    # model free reward config
    "punish_sur_mode": "max",
    "enable_slow_reward": True,
    "R_step": 12.0,
    "P_lat": 7.5,
    "P_long": 8.0,
    "P_phi": 2.0,
    "P_yaw": 1.5,
    "P_front": 5.0,
    "P_side": 0.0,
    "P_space": 5.0,
    "P_rear": 0.0,
    "P_steer": 0.15,
    "P_acc": 0.2,
    "P_delta_steer": 0.25,
    "P_jerk": 0.3,
    "P_done": 200.0,
    "P_boundary": 0,
    "safety_lat_margin_front": 0.3,
    "safety_long_margin_front": 0.0,
    "safety_long_margin_side": 2.0,
    "front_dist_thd": 50.0,
    "space_dist_thd": 12.0,
    "rel_v_thd": 1.0,
    "rel_v_rear_thd": 3.0,
    "time_dist": 0.5,
}

model_config_base = {
    "N": pre_horizon,
    "sur_obs_padding": "rule",
    "add_boundary_obs": True,
    "full_horizon_sur_obs": False,
    "ahead_lane_length_min": 6.0,
    "ahead_lane_length_max": 60.0,
    "v_discount_in_junction_straight": 0.75,
    "v_discount_in_junction_left_turn": 0.3,
    "v_discount_in_junction_right_turn": 0.3,
    "num_ref_lines": 3,
    "dec_before_junction_green": 0.8,
    "dec_before_junction_red": 1.3,
    "ego_length": 5.0,
    "ego_width": 1.8,
    "safe_dist_incremental": 1.2,
    "downsample_ref_point_index": (0, 1, 10,15,20, 30),

    "num_ref_points": pre_horizon + 1,
    "ego_feat_dim": 7,  # vx, vy, r, last_last_acc, last_last_steer, last_acc, last_steer
    "ego_bound_dim": 2,  # left, right
    "per_sur_state_dim": 6,  # x, y, phi, speed, length, width
    "per_sur_state_withinfo_dim": 7,  # x, y, phi, speed, length, width, mask
    "per_sur_feat_dim": 5,  # x, y, cos(phi), sin(phi), speed
    "per_ref_feat_dim": 5,  # x, y, cos(phi), sin(phi), speed
    "real_action_upper": (0.8, 0.571),
    "real_action_lower": (-3.0, -0.571),
    "steer_rate_2_min": -0.2,
    "steer_rate_2_max": 0.2,

    "vx_min": 0.0,
    "vx_max": 20.0,
    "vy_min": -4.0,
    "vy_max": 4.0,

    "max_dist_from_ref": 1.8,

    "Q": (
        0.4,
        0.4,
        500.0,
        1.0,
        2.0,
        300.0,
    ),
    "R": (
        1.0,
        20.0,
    ),

    "C_acc_rate_1": 0.0,
    "C_steer_rate_1": 10.0,
    "C_steer_rate_2": (10.0, 10.0), # C_steer_rate_2_min, C_steer_rate_2_max
    "C_v": (100., 100., 100., 100.), # C_vx_min, C_vx_max, C_vy_min, C_vy_max

    "gamma": 1.0,  # should equal to discount factor of algorithm
    "lambda_c": 0.99,  # discount of lat penalty
    "lambda_p": 0.99,  # discount of lon penalty
    "C_lat": 3.0,
    "C_obs": 300.0,
    "C_back": (
        0.1,  # surr is behind ego
        1.0  # surr is in front of ego
    ),
    "C_road": 300.0,
    "ref_v_lane": 12.0,
    "filter_num": 5
}

env_config_param_crossroad = env_config_param_base
model_config_crossroad = model_config_base

env_config_param_multilane = {
    **env_config_param_base,
    "scenario_root": MAP_ROOT_MULTILANE,
    "action_lower_bound": (-2.5 * delta_t*2, -0.065 * delta_t*2),
    "action_upper_bound": (2.5 * delta_t*2, 0.065 * delta_t*2),
    "real_action_lower_bound": (-1.5, -0.065),
    "real_action_upper_bound": (0.8, 0.065),
    "use_random_acc": True,
    "random_acc_cooldown": (30, 50, 50), # cooldown for acceleration, deceleration and ref_v, respectively
    "random_acc_prob": (0.1, 0.1), # probability to accelerate and decelerate, respectively
    "random_acc_range": (0.2, 0.8), # (m/s^2), used for acceleration
    "random_dec_range": (-1.5, -0.5), # (m/s^2), used for deceleration
}

model_config_multilane = {
    **model_config_base,
    "real_action_lower": (-1.5, -0.065),
    "real_action_upper": (0.8, 0.065),
    "Q": (
        0.,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
    "R": (
        0.0,
        0.0,
    ),
    # "reward_comps": ( 
    #     "env_pun2front",
    #     "env_pun2side",
    #     "env_pun2space",
    #     "env_pun2rear",
    #     "env_reward_vel_long",
    #     "env_reward_yaw_rate",
    #     "env_reward_dist_lat",
    #     "env_reward_head_ang",
    #     "env_reward_steering",
    #     "env_reward_acc_long",
    #     "env_reward_delta_steer",
    #     "env_reward_jerk",
    # )
    "reward_comps": ( 
            "env_scaled_reward_part2",
            "env_scaled_reward_step",
            "env_scaled_reward_dist_lat",
            "env_scaled_reward_vel_long",
            "env_scaled_reward_head_ang",
            "env_scaled_reward_yaw_rate",
            "env_scaled_reward_steering",
            "env_scaled_reward_acc_long",
            "env_scaled_reward_delta_steer",
            "env_scaled_reward_jerk",
            "env_scaled_reward_part1",
            "env_reward_collision_risk",
            "env_scaled_pun2front",
            "env_scaled_pun2side",
            "env_scaled_pun2space",
            "env_scaled_pun2rear",
            "env_scaled_punish_boundary",
            "env_scaled_reward_done",
            "env_scaled_reward_collision",
            )
}


def get_idsim_env_config(scenario="crossroad") -> Dict:
    if scenario == "crossroad":
        return env_config_param_crossroad
    elif scenario == "multilane":
        return env_config_param_multilane
    else:
        raise NotImplementedError


def get_idsim_model_config(scenario="crossroad") -> Dict:
    if scenario == "crossroad":
        return model_config_crossroad
    elif scenario == "multilane":
        return model_config_multilane
    else:
        raise NotImplementedError
    
def cal_idsim_obs_scale(
        ego_scale: Union[float, list] = 1.0,
        sur_scale: Union[float, list] = 1.0,
        ref_scale: Union[float, list] = 1.0,
        bound_scale: Union[float, list] = 1.0,
        env_config: Dict = None,
        env_model_config: Dict = None,
):
    ego_dim = env_model_config["ego_feat_dim"]
    sur_dim = env_model_config["per_sur_feat_dim"] + 3 # +3 for length, width, mask
    ref_dim = env_model_config["per_ref_feat_dim"]
    bound_dim = env_model_config["ego_bound_dim"]
    sur_num = int(sum(i for i in env_config["obs_num_surrounding_vehicles"].values()))
    full_horizon_sur_obs = env_model_config["full_horizon_sur_obs"]
    num_ref_points = len(env_model_config["downsample_ref_point_index"]) 

    if isinstance (ego_scale, float):
        ego_scale = [ego_scale] * ego_dim
    if isinstance (sur_scale, float):
        sur_scale = [sur_scale] * sur_dim
    if isinstance (ref_scale, float):
        ref_scale = [ref_scale] * ref_dim
    if isinstance (bound_scale, float):
        bound_scale = [bound_scale] * bound_dim

    assert len(ego_scale) == ego_dim, f"len(ego_scale)={len(ego_scale)}, ego_dim={ego_dim}"
    assert len(sur_scale) == sur_dim, f"len(sur_scale)={len(sur_scale)}, sur_dim={sur_dim}"
    assert len(ref_scale) == ref_dim, f"len(ref_scale)={len(ref_scale)}, ref_dim={ref_dim}"
    assert len(bound_scale) == bound_dim, f"len(boundary_scale)={len(bound_scale)}, bound_dim={bound_dim}"
    
    obs_scale = []
    obs_scale += ego_scale

    for scale in ref_scale:
        obs_scale += [scale] * num_ref_points

    if full_horizon_sur_obs:
        obs_scale += (sur_scale * sur_num * num_ref_points)
    else:
        obs_scale += sur_scale * sur_num
    if env_model_config["add_boundary_obs"]:
        obs_scale += bound_scale

    obs_scale = np.array(obs_scale, dtype=np.float32)
    return obs_scale

def cal_idsim_pi_paras(
        env_config: Dict = None,
        env_model_config: Dict = None,
):
    ego_dim = env_model_config["ego_feat_dim"]
    sur_dim = env_model_config["per_sur_feat_dim"] + 3 # +3 for length, width, mask
    ref_dim = env_model_config["per_ref_feat_dim"]
    num_ref_points = len(env_model_config["downsample_ref_point_index"]) 
    num_objs = int(sum(i for i in env_config["obs_num_surrounding_vehicles"].values()))

    pi_paras = {}
    pi_paras["pi_begin"] = ego_dim + ref_dim*num_ref_points
    pi_paras["pi_end"] = pi_paras["pi_begin"] + sur_dim*num_objs 
    pi_paras["obj_dim"] = sur_dim 
    pi_paras["output_dim"] = sur_dim*num_objs + 1
    return pi_paras

