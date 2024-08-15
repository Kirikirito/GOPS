import os
import sys
import datetime
import pathlib
import json
import pickle
import torch
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing
from dataclasses import dataclass
from typing import NamedTuple, Optional
from multiprocessing import Pool

from gops.trainer.idsim_idc_mf_multilane_evaluator import (
    IdsimIDCEvaluator, EvalResult,
    get_args_from_json,
)
from gops.trainer.idsim_render.animation_mf_multilane import AnimationLane
from gops.trainer.idsim_render.animation_crossroad import AnimationCross

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def parallel_eval(configs_dict, config_file_path= None):
    theme_style = "light"  # only works for AnimationCross
    args = get_args(configs_dict, config_file_path)
    test_case_list = load_test_case(args["test_case_file"], args["test_filter"])
    render_only = args["render_only"]

    # Prepare the multiprocessing pool
    num_processes = multiprocessing.cpu_count()
    pool = Pool(processes=num_processes)
    if not render_only:
            # Parallel processing for test cases
        test_case_results = pool.starmap(run_test_case, [(args, idx, test_case) for idx, test_case in enumerate(test_case_list)])

    # Load the configuration file
    save_folder = args["save_folder"]
    log_path_root = pathlib.Path(save_folder)
    config_path = log_path_root / "test_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    if args["generate_animation"]:
        # Parallel processing for animations
        pool.starmap(generate_animation, [(args, idx, test_case, config, log_path_root, theme_style) for idx, test_case in enumerate(test_case_list)])
    pool.close()
    pool.join()
    print("All test cases are done.")
    print("test case results: ", test_case_results)


def multi_parallel_eval(configs_dict_list, config_file_path= None):
    # for configs_dict in configs_dict_list:
    #     parallel_eval(configs_dict, config_file_path)
    theme_style = "light"  # only works for AnimationCross
    args_list = [get_args(configs_dict, config_file_path) for configs_dict in configs_dict_list]
    test_case_lists = [load_test_case(args["test_case_file"], args["test_filter"]) for args in args_list]
    render_only = args_list[0]["render_only"] # NOTE: all the args should have the same render_only value


    # Prepare the multiprocessing pool
    num_processes = multiprocessing.cpu_count()
    pool = Pool(processes=num_processes)
    if not render_only:
        # Parallel processing for test cases
        test_case_results = pool.starmap(run_test_case, [(args, idx, test_case) for args, test_case_list in zip(args_list, test_case_lists) for idx, test_case in enumerate(test_case_list)])
    
    # Load the configuration file
    save_folders = [args["save_folder"] for args in args_list]
    log_path_roots = [pathlib.Path(save_folder) for save_folder in save_folders]
    config_paths = [log_path_root / "test_config.json" for log_path_root in log_path_roots]
    configs = [json.load(open(config_path, "r")) for config_path in config_paths]
    if args_list[0]["generate_animation"]: # NOTE: all the args should have the same generate_animation value
        # Parallel processing for animations
        pool.starmap(generate_animation, [(args, idx, test_case, config, log_path_root, theme_style) for args, test_case_list, config, log_path_root in zip(args_list, test_case_lists, configs, log_path_roots) for idx, test_case in enumerate(test_case_list)])
    pool.close()
    pool.join()
    print("All test cases are done.")
    smo_dict_list = []
    for idx, args,config_dict, test_case_list, log_path_root in zip(range(len(args_list)), args_list,configs_dict_list, test_case_lists, log_path_roots):
        smo_dict = {}
        smo_dict["network"] = config_dict["ini_network_root"]
        smo_dict["iteration"] = config_dict["nn_index"]
        for idx, test_case in enumerate(test_case_list):
            smo_dict.update(cal_action_smoothness(idx, test_case, log_path_root))
            smo_dict["noise_data"] = args["obs_noise_data"]
            smo_dict["noise_type"] = args["obs_noise_type"]
            smo_dict["noise_level"]= config_dict["obs_noise_level"]
            smo_dict_list.append(smo_dict)
    plot_smoothness(smo_dict_list)
    return smo_dict_list

def plot_smoothness(smo_dict_list):
    smo_df = pd.DataFrame(smo_dict_list)
    for network_root in smo_df["network"].unique():
        network_df = smo_df[smo_df["network"] == network_root]
        title = f"{network_root.split('/')[-2]}_smoothness"
        plt.figure()
        sns.lineplot(data=network_df, x="noise_level", y="real_afr", hue="iteration")
        plt.title(f"{title} real afr")
        plt.savefig(f"{network_root}_real_afr.png")
        plt.figure()
        sns.lineplot(data=network_df, x="noise_level", y="real_sdv", hue="iteration")
        plt.title(f"{title} real sdv")
        plt.savefig(f"{network_root}_real_sdv.png")
        plt.figure()
        sns.lineplot(data=network_df, x="noise_level", y="real_mwf", hue="iteration")
        plt.title(f"{title} real mwf")
        plt.savefig(f"{network_root}_real_mwf.png")
        plt.figure()
        sns.lineplot(data=network_df, x="noise_level", y="inc_afr", hue="iteration")
        plt.title(f"{title} inc afr")
        plt.savefig(f"{network_root}_inc_afr.png")
        plt.figure()
        sns.lineplot(data=network_df, x="noise_level", y="inc_sdv", hue="iteration")
        plt.title(f"{title} inc sdv")
        plt.savefig(f"{network_root}_inc_sdv.png")
        plt.figure()
        sns.lineplot(data=network_df, x="noise_level", y="inc_mwf", hue="iteration")
        plt.title(f"{title} inc mwf")
        plt.savefig(f"{network_root}_inc_mwf.png")


def combine_conf(test_conf, conf_list):
    test_conf_list = []
    for ini_network_root, nn_index in zip(conf_list.ini_network_root, conf_list.nn_index):
        for idx, noise in enumerate(conf_list.noise_data):
            test_conf_copy = test_conf.copy()
            test_conf_copy["ini_network_root"] = ini_network_root
            test_conf_copy["nn_index"] = nn_index
            test_conf_copy["noise_data"] = noise
            test_conf_copy["test_name"] = test_conf["test_name"] + "_" + nn_index + "_noise_level_" + str(idx)
            test_conf_copy["obs_noise_level"] = idx
            test_conf_list.append(test_conf_copy)
    return test_conf_list

def generate_noise_data(noise_type, num=10, scale='log', low=1e-3, high=1e-1):
    if noise_type == 'uniform' or noise_type == 'normal':
        if scale == 'log':
            noise_std = np.logspace(np.log10(low), np.log10(high), num)
        elif scale == 'linear':
            noise_std = np.linspace(low, high, num)
        else:
            raise ValueError("scale should be log or linear")
        noise_data = [[0, noise] for noise in noise_std]
    elif noise_type == 'sine':
        raise NotImplementedError
    return noise_data
def find_optimal_network(ini_network_root):
    # find the netowrk end with _opt.pkl if not found, return the lastest network
    appr_dir = os.path.join(ini_network_root, 'apprfunc')
    network_list = []
    for file in os.listdir(appr_dir):
        if file.endswith('_opt.pkl'):
            network_list.append(file)
    if len(network_list) > 0:
        network_list.sort()
        index , surfix = network_list[-1].split('_')[1:3]
        surfix = surfix.split('.')[0]
        index = index + '_' + surfix
    if len(network_list) == 0:
        network_list = [file for file in os.listdir(appr_dir) if file.endswith('.pkl')]
        network_list.sort()
        index = network_list[-1].split('_')[1]
        index = index.split('.')[0]
    return index


def load_test_case(test_case_path="example.json", filter = None):
    # load test case
    with open(test_case_path, "r", encoding="utf-8") as file:
        test_case_list = json.load(file)
    # filter test case
    if filter is not None:
        for key, value in filter.items():
            if isinstance(value, str):
                test_case_list = [test_case for test_case in test_case_list if re.match(value, test_case.get(key, ""))]
            elif isinstance(value, int):
                test_case_list = [test_case for test_case in test_case_list if test_case.get(key, 0) > value]
    for test in test_case_list:
        print(f"test case: {test['scene']} map: {test['map_id']} warmup time: {test['warmup_time']}") 

    return test_case_list

def change_type(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = change_type(value)
        return obj
    elif isinstance(obj, list):
        return [change_type(value) for value in obj]
    elif isinstance(obj, pathlib.Path):
        return str(obj)
    else:
        return obj
    

def cal_noise_mask(args):
    env_model_config = args["env_model_config"] 
    env_config = args["env_config"]
    ego_scale = args["ego_scale"]
    sur_scale = args["sur_scale"]
    ref_scale = args["ref_scale"]
    bound_scale = 1.0
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
    sur_scale[-1] = 0.0   # HACK: mask should not be noisy a tricky way to do this
    obs_scale_with_mask = []
    obs_scale_with_mask += ego_scale

    for scale in ref_scale:
        obs_scale_with_mask += [scale] * num_ref_points

    if full_horizon_sur_obs:
        obs_scale_with_mask += (sur_scale * sur_num * num_ref_points)
    else:
        obs_scale_with_mask += sur_scale * sur_num
    if env_model_config["add_boundary_obs"]:
        obs_scale_with_mask += bound_scale

    obs_scale_with_mask = np.array(obs_scale_with_mask, dtype=np.float32)
    return obs_scale_with_mask.copy()

def run_test_case(args, idx, test_case):
    IDCevaluator = IdsimIDCEvaluator(**args)
    idsim_tb_eval_dict = IDCevaluator.run_testcase(idx, test_case, use_mpc=False)
    IDCevaluator.env.close()
    return idsim_tb_eval_dict

def generate_animation(args, idx, test_case, config, log_path_root, theme_style):
    map_id = int(test_case["map_id"])
    test_scene = test_case.get("scene", "scene" + str(idx))
    suffix = "%03d" % map_id
    log_path = log_path_root / f"test_{idx}" / suffix

    episode_data = get_episode_data(idx, test_case, log_path_root)
    fcd_file_path = log_path / "fcd.xml"
    if args["env_scenario"] == "crossroad":
        animation = AnimationCross(theme_style, fcd_file_path, config)
    else:
        animation = AnimationLane(theme_style, fcd_file_path, config)

    animation.generate_animation(episode_data, log_path_root, idx, test_scene=test_scene, mode="debug", dpi=args["dpi"], frame_skip=args["frame_skip"], plot_reward=args["plot_reward"])


def get_episode_data(idx, test_case, log_path_root) -> EvalResult:
    map_id = int(test_case["map_id"])
    suffix = "%03d" % map_id
    log_path = log_path_root / f"test_{idx}" / suffix

    episode_list = [
        path
        for path in os.listdir(log_path)
        if (path.startswith("episode") and path.endswith(".pkl"))
    ]
    if len(episode_list) != 1:
        raise ValueError("episode_list should only have one episode")
    episode = episode_list[0]

    episode_path = log_path / episode
    with open(episode_path, "rb") as f:
        episode_data = pickle.load(f)
    return episode_data

def cal_action_smoothness(idx, test_case, log_path_root):
    episode_data = get_episode_data(idx, test_case, log_path_root)
    action_seq = episode_data.action_real_list
    afr = cal_afr(action_seq)
    sdv = cal_sdv(action_seq)
    mwf = cal_mwf(action_seq)
    print(f"Test case", test_case)
    print(f"real action smoothness: afr: {afr}, sdv: {sdv}, mwf: {mwf}")
    inc_action_seq = episode_data.action_list
    inc_afr = cal_afr(inc_action_seq)
    inc_sdv = cal_sdv(inc_action_seq)
    inc_mwf = cal_mwf(inc_action_seq)
    print(f"inc action smoothness: afr: {inc_afr}, sdv: {inc_sdv}, mwf: {inc_mwf}")
    smo_dict = {"real_afr": afr, "real_sdv": sdv, "real_mwf": mwf, "inc_afr": inc_afr, "inc_sdv": inc_sdv, "inc_mwf": inc_mwf}
    return smo_dict

def cal_afr(action_seq):
    action_seq = np.array(action_seq)
    daction_seq = action_seq[1:] - action_seq[:-1]
    afr = np.linalg.norm(daction_seq, axis=1).mean()
    return afr

def cal_sdv(action_seq):
    # calculate the second derivative variation
    action_seq = np.array(action_seq)
    daction_seq = action_seq[1:] - action_seq[:-1]
    ddaction_seq = daction_seq[1:] - daction_seq[:-1]
    sdv = np.linalg.norm(ddaction_seq, axis=1).mean()
    return sdv

def cal_mwf(action_seq):
# mean of the weighted frequency
    action_seq = np.array(action_seq)
    N = len(action_seq)
    dim = action_seq.shape[1]
    # sample spacing
    T = 0.1
    top = 0
    mwf = 0
    for i in range(dim):
        y = action_seq[:,i]
        yf = np.fft.fft(y)
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
        # remove first top points
        if top > 0:
            yf[:top] = 0
        # print("remove the five points with the highest amplitude")
        weighted_yf = np.sum(xf[top:N//2]*np.abs(yf[top:N//2]))/(np.sum(np.abs(yf[top:N//2]))+1e-8)
        avg_amp = np.sum(np.abs(yf[top:N//2]))/(N//2-top)
        mwf += weighted_yf
    mwf = mwf/dim
    return mwf
@dataclass
class EvalConf:
    render_only: bool = False
    generate_animation = True
    scenario: str = "multilane"
    num_scenarios: int = 34
    scenario_reuse: int = 1
    scenario_root: str = "/root/gops/gops/scenario/idsim" 
    ini_network_root: str = "/root/gops/gops/idsim"
    test_case_file: str = "/root/gops/gops/idsim/test_case.json"
    test_filter: dict = None
    nn_index: str = None
    dpi: int = 60
    frame_skip: int = 3
    tracking_only: bool = False
    noise_type: Optional[str] = None
    noise_data: Optional[list] = None
    rel_noise_scale: bool = False
    obs_noise_level: int = 0    # noise level for the test case
    test_name: str = 'test'
    scenario_filter_surrounding_selector: Optional[str] = None
    direction_selector: str = 's'
    IDC_mode: bool = False   # enable IDC mode
    fast_mode: bool = False  # do not recal value when druring cooldown save evaluation time
    multi_ref: bool = True   # use multiple reference path
    plot_reward: bool = False # plot reward in the animation
    random_ref_probability: float = 0.01 # probability to use random reference path
    path_selector: str = "value" # value or loss
    selector_bias: float = -2.5 # bias for current path
    max_steps: int = 2000 # max steps for each episode
    no_done_at_collision: bool = True

    # takeover bias config
    takeover_bias: bool = False
    bias_x: tuple = (0.0, 0.5)
    bias_y: tuple = (0.0, 0.5)
    bias_phi: tuple = (0.0, 0.05)
    bias_vx: tuple = (0.0, 1.5)
    bias_ax: tuple = (0.0, 0.25)
    bias_steer: tuple = (0.0, 0.02)

    minimum_clearance_when_takeover: float = 5.0

    # random ref_v config
    random_ref_v: bool = False
    ref_v_range: tuple = (0, 2)
    ref_v: float = 12
    use_random_acc: bool = False
    random_acc_prob: tuple = (0.3, 0.3, 1)
    random_acc_cooldown: tuple = (50, 50, 50)
    random_acc_range: tuple = (0.2, 0.8)
    random_dec_range: tuple = (0.2, 0.8)

    # sur_bias config
    sur_bias_range: tuple = (0, 0)
    sur_bias_prob: float = 0.0

    # env model config
    Q_mat: tuple = (0.0, 0.0, 0.0, 0.5, 0.0, 0.0)
    R_mat: tuple = (0.0, 0.0)
    C_acc_rate_1: float = 0.0
    C_steer_rate_1: float = 0.0
    C_steer_rate_2: tuple = (0.0, 0.0)

    def to_dict(self):
        return self.__dict__
    
    def from_dict(self, data: dict, strict: bool = True):
        for key, value in data.items():
            if not hasattr(self, key):
                if strict:
                    raise ValueError(f"Unknown key {key}")
                else:
                    continue
            setattr(self, key, value)
        return self
    
    def from_json(self, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        return self.from_dict(config)

    def to_json(self, config_path):
        # save the config to json file
        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        return config_path


def get_args_from_json(config_path, default_args):
    with open(config_path, "r") as f:
        config = json.load(f)
    for key, value in default_args.items():
        if key not in config:
            config[key] = value
    return config






def get_args(configs_dict, config_file_path= None):
    # args priority: configs_dict > config_file_path > default_args
    if config_file_path is not None:
        conf = EvalConf().from_json(config_file_path)
    else:
        conf = EvalConf()
    conf = conf.from_dict(configs_dict)
    if conf.nn_index is None:
        conf.nn_index = find_optimal_network(conf.ini_network_root)
    if conf.test_name is None:
        conf.test_name = datetime.datetime.now().strftime("%y%m%d-%H%M%S") + "IDCTest"
    save_folder = os.path.join(conf.ini_network_root, conf.test_name)
    if os.path.exists(save_folder) and not conf.render_only:
        val = input("Warning: save_folder already exists, press esc to exit, press enter to delete and continue")
        if val == chr(27):
            exit()
        else:
            # clear the content of save_folder remian the folder
            os.system("rm -rf {}/*".format(save_folder))
    else:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=False)

    # get training args
    config_path = os.path.join(conf.ini_network_root, "config.json")
    args = get_args_from_json(config_path, {})

    # save args to save_folder
    args["generate_animation"] = conf.generate_animation
    args["render_only"] = conf.render_only
    args["save_folder"] = save_folder
    args["test_case_file"] = conf.test_case_file
    args['env_scenario'] = conf.scenario
    args["test_filter"] = conf.test_filter
    args["plot_reward"] = conf.plot_reward
    args["IDC_MODE"] = conf.IDC_mode
    args["fast_mode"] =   conf.fast_mode
    args["PATH_SELECTION_EVIDENCE"] = conf.path_selector
    args["PATH_SELECTION_DIFF_THRESHOLD"] = conf.selector_bias
    args["frame_skip"] =    conf.frame_skip
    args["dpi"] = conf.dpi
    args["nn_index"] = conf.nn_index # NOTE: not in the config.json
    args["ini_network_root"] = conf.ini_network_root
    args["ini_network_dir"] = conf.ini_network_root + f"/apprfunc/apprfunc_{conf.nn_index}.pkl"
    args["obs_noise_type"] = conf.noise_type
    args["obs_noise_data"] = conf.noise_data
    args["rel_noise_scale"] = conf.rel_noise_scale

    args["env_config"]["max_steps"] = conf.max_steps
    args["env_config"]["use_multiple_path_for_multilane"] = conf.multi_ref
    args["env_config"]["random_ref_probability"] = conf.random_ref_probability
    args["env_config"]["scenario_root"] = pathlib.Path(conf.scenario_root)
    args["env_config"]["direction_selector"] = conf.direction_selector
    args["env_config"]["no_done_at_collision"] = conf.no_done_at_collision
    args["env_config"]["scenario_filter_surrounding_selector"] = conf.scenario_filter_surrounding_selector
    args["env_config"]["ignore_surrounding"] = conf.tracking_only

    args["env_config"]["takeover_bias"] = conf.takeover_bias
    args["env_config"]["takeover_bias_x"] = conf.bias_x
    args["env_config"]["takeover_bias_y"] = conf.bias_y
    args["env_config"]["takeover_bias_phi"] = conf.bias_phi
    args["env_config"]["takeover_bias_vx"] = conf.bias_vx
    args["env_config"]["takeover_bias_ax"] = conf.bias_ax
    args["env_config"]["takeover_bias_steer"] = conf.bias_steer
    args["env_config"]["minimum_clearance_when_takeover"] = conf.minimum_clearance_when_takeover

    args["env_config"]["sur_bias_range"] = conf.sur_bias_range
    args["env_config"]["sur_bias_prob"] = conf.sur_bias_prob


    args["env_config"]["random_ref_v"] = conf.random_ref_v
    args["env_config"]["ref_v_range"] = conf.ref_v_range
    args["env_config"]["ref_v"] = conf.ref_v
    args["env_model_config"]["ref_v_lane"] = conf.ref_v

    args["env_config"]["use_random_acc"] = conf.use_random_acc
    args["env_config"]["random_acc_prob"] = conf.random_acc_prob
    args["env_config"]["random_acc_cooldown"] = conf.random_acc_cooldown
    args["env_config"]["random_acc_range"] = conf.random_acc_range
    args["env_config"]["random_dec_range"] = conf.random_dec_range

    args["env_model_config"]["Q"] = conf.Q_mat
    args["env_model_config"]["R"] = conf.R_mat
    args["env_model_config"]["C_acc_rate_1"] = conf.C_acc_rate_1
    args["env_model_config"]["C_steer_rate_1"] = conf.C_steer_rate_1
    args["env_model_config"]["C_steer_rate_2"] = conf.C_steer_rate_2

    args["eval_save"] = True
    args["repeat_num"] = None
    args["record_loss"] = True

    args["env_config"]["use_render"] = False
    args["env_config"]["num_scenarios"] = conf.num_scenarios
    args["env_config"]["scenario_reuse"] = conf.scenario_reuse
    args["env_config"]["use_logging"] = True
    args["env_config"]["singleton_mode"] = "invalidate"
    args["env_config"]["logging_name_template"] = "{context.scenario_id:03d}/{context.episode_count:04d}.pkl"
    args["env_config"]["fcd_name_template"] = "{context.scenario_id:03d}/fcd.xml"

    with open(os.path.join(save_folder, "test_config.json"), "w") as f:
        json.dump(args, f, indent=4, default= change_type)
    os.chmod(os.path.join(save_folder, "test_config.json"), 0o444) # read only
    return args