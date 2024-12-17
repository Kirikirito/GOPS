#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Utils Function
#  Update Date: 2021-03-10, Yuhang Zhang: Create codes


import sys
import os
import torch.nn as nn
import numpy as np
import logging
import time
from typing import Optional

from gops.utils.act_distribution_type import *
import random

logger = logging.getLogger(__name__)


def get_activation_func(key: str):
    assert isinstance(key, str)

    activation_func = None
    if key == "relu":
        activation_func = nn.ReLU

    elif key == "elu":
        activation_func = nn.ELU

    elif key == "gelu":
        activation_func = nn.GELU

    elif key == "selu":
        activation_func = nn.SELU

    elif key == "sigmoid":
        activation_func = nn.Sigmoid

    elif key == "tanh":
        activation_func = nn.Tanh

    elif key == "linear":
        activation_func = nn.Identity
    
    elif key == "softplus":
        activation_func = nn.Softplus   

    if activation_func is None:
        print("input activation name:" + key)
        raise RuntimeError

    return activation_func


def get_apprfunc_dict(key: str, **kwargs):
    var = dict()
    var["apprfunc"] = kwargs[key + "_func_type"]
    var["name"] = kwargs[key + "_func_name"]
    var["obs_dim"] = kwargs["obsv_dim"]
    var["min_log_std"] = kwargs.get(key + "_min_log_std", float("-20"))
    var["max_log_std"] = kwargs.get(key + "_max_log_std", float("2"))
    var["std_type"] = kwargs.get(key + "_std_type", "mlp_shared")
    var["norm_matrix"] = kwargs.get("norm_matrix", None)
    var["pre_horizon"] = kwargs.get("pre_horizon", None)
    var["additional_info"] = kwargs.get("additional_info", None)

    apprfunc_type = kwargs[key + "_func_type"]
    if key + "_output_activation" not in kwargs.keys():
        kwargs[key + "_output_activation"] = "linear"
    if apprfunc_type == "MLP":
        var["hidden_sizes"] = kwargs[key + "_hidden_sizes"]
        var["hidden_activation"] = kwargs[key + "_hidden_activation"]
        var["output_activation"] = kwargs[key + "_output_activation"]
    elif apprfunc_type == "RNN":
        var["hidden_sizes"] = kwargs[key + "_hidden_sizes"]
        var["hidden_activation"] = kwargs[key + "_hidden_activation"]
        var["output_activation"] = kwargs[key + "_output_activation"]
        var["rnn_type"] = kwargs[key + "_rnn_type"]
        var["num_layers"] = kwargs[key + "_num_layers"]
    elif apprfunc_type == "GAUSS":
        var["num_kernel"] = kwargs[key + "_num_kernel"]
    elif apprfunc_type == "CNN":
        var["hidden_activation"] = kwargs[key + "_hidden_activation"]
        var["output_activation"] = kwargs[key + "_output_activation"]
        var["conv_type"] = kwargs[key + "_conv_type"]
    elif apprfunc_type == "CNN_SHARED":
        if key == "feature":
            var["conv_type"] = kwargs["conv_type"]
        else:
            var["feature_net"] = kwargs["feature_net"]
            var["hidden_activation"] = kwargs[key + "_hidden_activation"]
            var["output_activation"] = kwargs[key + "_output_activation"]
    elif apprfunc_type == "POLY":
        var["degree"] = kwargs[key + "_degree"]
        var["add_bias"] = kwargs[key + "_add_bias"]
    elif apprfunc_type == "GAUSS":
        var["num_kernel"] = kwargs[key + "_num_kernel"]
    elif apprfunc_type == "Attention":
        var["hidden_sizes"] = kwargs[key + "_hidden_sizes"]
        var["hidden_activation"] = kwargs[key + "_hidden_activation"]
        var["output_activation"] = kwargs[key + "_output_activation"]
        var["attn_in_per_dim"] = kwargs["attn_in_per_dim"]     
        var["attn_out_dim"] = kwargs["attn_out_dim"]
        var["attn_begin"] = kwargs["attn_begin"]
        var["attn_end"] = kwargs["attn_end"]
    elif apprfunc_type == "LipsNet":
        var["hidden_sizes"] = kwargs[key + "_hidden_sizes"]
        var["hidden_activation"] = kwargs[key + "_hidden_activation"]
        var["output_activation"] = kwargs[key + "_output_activation"]
        var["lips_init_value"] = kwargs[key + "_lips_init_value"]
        var["lips_auto_adjust"] = kwargs[key + "_lips_auto_adjust"]
        var["lips_learning_rate"] = kwargs[key + "_lips_learning_rate"]
        var["lips_hidden_sizes"] = kwargs[key + "_lips_hidden_sizes"]
        var["eps"] = kwargs[key + "_eps"]
        var["lambda"] = kwargs[key + "_lambda"]
        var["local_lips"] = kwargs[key + "_local_lips"]
        var["squash_action"] = kwargs[key + "_squash_action"]
        var["learning_rate"] = kwargs[key + "_learning_rate"]
    elif apprfunc_type == "PINet":
        if key == "pi_net":
            var["pi_begin"] = kwargs["pi_begin"]
            var["pi_end"] = kwargs["pi_end"]
            var["enable_mask"] = kwargs["enable_mask"]
            var["obj_dim"] = kwargs["obj_dim"]
            var["pi_out_dim"] = kwargs["pi_out_dim"]
            var["encoding_others"] = kwargs["encoding_others"]
            var["enable_self_attention"] = kwargs["enable_self_attention"]
            var["use_multi_head"] = kwargs.get("use_multi_head", False)
            var["head_num"] = kwargs.get("head_num", None)
            var["attn_dim"] = kwargs.get("attn_dim", None)
            if var["encoding_others"]:
                var["others_out_dim"] = kwargs["others_out_dim"]
                var["others_hidden_sizes"] = kwargs["others_hidden_sizes"]
                var["others_hidden_activation"] = kwargs["others_hidden_activation"]
                var["others_output_activation"] = kwargs["others_output_activation"]
            var["pi_hidden_sizes"] = kwargs["pi_hidden_sizes"]
            var["pi_hidden_activation"] = kwargs["pi_hidden_activation"]
            var["pi_output_activation"] = kwargs["pi_output_activation"]
        else:
            var["pi_net"] = kwargs["pi_net"]
            var["target_PI"] = kwargs["target_PI"]
            var["freeze_pi_net"] = kwargs["freeze_pi_net"]
            assert var["freeze_pi_net"] in ["actor", "critic", "none","both"]
            var["hidden_sizes"] = kwargs[key + "_hidden_sizes"]
            var["hidden_activation"] = kwargs[key + "_hidden_activation"]
            var["output_activation"] = kwargs[key + "_output_activation"]
    elif apprfunc_type == "SMONET" or apprfunc_type == "SMONET0" or apprfunc_type == "SMONET2" or apprfunc_type == "SMONET3" or apprfunc_type == "SMONET4" or apprfunc_type == "SMONET5" or apprfunc_type == "SMONET6":
        var["hidden_sizes"] = kwargs[key + "_hidden_sizes"]
        var["hidden_activation"] = kwargs[key + "_hidden_activation"]
        var["output_activation"] = kwargs[key + "_output_activation"]
        var["kernel_size"] = kwargs[key + "_kernel_size"]
        var["loss_weight"] = kwargs[key + "_loss_weight"]
        var["tau_layer_num"] = kwargs["tau_layer_num"]
        var["seq_len"] = kwargs["seq_len"]
    elif apprfunc_type == "PISMONET":
        if key == "pi_net":
            var["pi_begin"] = kwargs["pi_begin"]
            var["pi_end"] = kwargs["pi_end"]
            var["enable_mask"] = kwargs["enable_mask"]
            var["obj_dim"] = kwargs["obj_dim"]
            var["pi_out_dim"] = kwargs["pi_out_dim"]
            var["encoding_others"] = kwargs["encoding_others"]
            var["enable_self_attention"] = kwargs["enable_self_attention"]
            var["use_multi_head"] = kwargs.get("use_multi_head", False)
            var["head_num"] = kwargs.get("head_num", None)
            var["attn_dim"] = kwargs.get("attn_dim", None)
            if var["encoding_others"]:
                var["others_out_dim"] = kwargs["others_out_dim"]
                var["others_hidden_sizes"] = kwargs["others_hidden_sizes"]
                var["others_hidden_activation"] = kwargs["others_hidden_activation"]
                var["others_output_activation"] = kwargs["others_output_activation"]
            var["pi_hidden_sizes"] = kwargs["pi_hidden_sizes"]
            var["pi_hidden_activation"] = kwargs["pi_hidden_activation"]
            var["pi_output_activation"] = kwargs["pi_output_activation"]
        else:
            var["pi_net"] = kwargs["pi_net"]
            var["target_PI"] = kwargs["target_PI"]
            var["freeze_pi_net"] = kwargs["freeze_pi_net"]
            assert var["freeze_pi_net"] in ["actor", "critic", "none","both"]
            var["hidden_sizes"] = kwargs[key + "_hidden_sizes"]
            var["hidden_activation"] = kwargs[key + "_hidden_activation"]
            var["output_activation"] = kwargs[key + "_output_activation"]
            var["kernel_size"] = kwargs[key + "_kernel_size"]
            var["loss_weight"] = kwargs[key + "_loss_weight"]
            var["tau_layer_num"] = kwargs["tau_layer_num"]
            var["seq_len"] = kwargs["seq_len"]
    elif apprfunc_type == "MLPA":
        var["hidden_sizes"] = kwargs[key + "_hidden_sizes"]
        var["hidden_activation"] = kwargs[key + "_hidden_activation"]
        var["output_activation"] = kwargs[key + "_output_activation"]
        var["adapter_layers"] = kwargs[key + "_adapter_layers"]
        var["consid_last_act"] = kwargs[key + "_consid_last_act"]
        var["adapter_start_layer"] = kwargs[key + "_adapter_start_layer"]
        var["adapter_add_bias"] = kwargs[key + "_adapter_add_bias"]
        var["adaption_ratio"] = kwargs[key + "_adaption_ratio"]
    else:
        raise NotImplementedError

    if kwargs["action_type"] == "continu":
        var["act_high_lim"] = np.array(kwargs["action_high_limit"])
        var["act_low_lim"] = np.array(kwargs["action_low_limit"])
        var["act_dim"] = kwargs["action_dim"]

    else:
        var["act_num"] = kwargs["action_num"]

    if kwargs["policy_act_distribution"] == "default":
        if kwargs["action_type"] == "continu":
            if kwargs["policy_func_name"] == "StochaPolicy":  # todo: add TanhGauss
                var["action_distribution_cls"] = GaussDistribution
            elif kwargs["policy_func_name"] == "DetermPolicy" or "FiniteHorizonPolicy" or "AttentionPolicy" or "AttentionFullPolicy":
                var["action_distribution_cls"] = DiracDistribution
        else:
            if kwargs["policy_func_name"] == "StochaPolicyDis":
                var["action_distribution_cls"] = CategoricalDistribution
            elif kwargs["policy_func_name"] == "DetermPolicyDis":
                var["action_distribution_cls"] = ValueDiracDistribution
    else:

        var["action_distribution_cls"] = getattr(
            sys.modules[__name__], kwargs["policy_act_distribution"]
        )

    return var


def change_type(obj):
    if isinstance(
        obj,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)
    elif isinstance(obj, type):
        return str(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = change_type(v)
        return obj
    elif isinstance(obj, list):
        for i, o in enumerate(obj):
            obj[i] = change_type(o)
        return obj
    else:
        return obj


def random_choice_with_index(obj_list):
    obj_len = len(obj_list)
    random_index = random.choice(list(range(obj_len)))
    random_value = obj_list[random_index]
    return random_value, random_index


def array_to_scalar(arrayLike):
    """Convert size-1 array to scalar"""
    return arrayLike if isinstance(arrayLike, (int, float)) else arrayLike.item()


def seed_everything(seed: Optional[int] = None) -> int:
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if seed is None:
        seed = random.randint(min_seed_value, max_seed_value)

    elif not isinstance(seed, int):
        # noinspection PyTypeChecker
        seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed

def cal_ave_exec_time(print_interval=100):
    def decorator(func):
        total_time = 0
        count = 0

        def wrapper(*args, **kwargs):
            nonlocal total_time, count
            start_time = time.perf_counter_ns()
            result = func(*args, **kwargs)
            end_time = time.perf_counter_ns()
            execution_time = (end_time - start_time) / 1e6
            total_time += execution_time
            count += 1
            if count % print_interval == 0:
                print(f"#INFO: {func.__name__}: average execution time after {count} steps: {total_time / count:.9f} ms")
            return result
        return wrapper
    return decorator


def set_seed(trainer_name, seed, offset, env=None):
    """
    When trainer_name is `**_async_**` or `**_sync_**`, set random seed for subprocess and gym env,
    else only set subprocess for gym env

    Parameters
    ----------
    trainer_name : str
        trainer_name
    seed : int
        global seed
    offset : int
        offset of random seed for  subprocess
    env : gym.Env, optional
        gym env needs to set random seed, by default None

    Returns
    -------
    (int, gym.Env)
        random seed for subprocess, gym env which random seed is set
    """

    if trainer_name.split("_")[1] in ["async", "sync"]:
        print("Setting seed of a subprocess to {}".format(seed + offset))
        seed_everything(seed + offset)
        if env is not None:
            env.seed(seed + offset)
        return seed + offset, env

    else:
        if env is not None:
            env.seed(seed)
        return None, env


class FreezeParameters:
    def __init__(self, modules, freeze=True):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.freeze = freeze
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        if self.freeze:
            for param in get_parameters(self.modules):
                param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.freeze:
            for i, param in enumerate(get_parameters(self.modules)):
                param.requires_grad = self.param_states[i]


def get_parameters(modules):
    """
    Given list of torch modules, returns list of their parameters.
    :param modules: iterable of modules
    :returns: list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters

class TimePerfRecorder:
    def __init__(self, context, print_interval=100):
        self.context = context
        self.total_time = 0
        self.count = 0
        self.print_interval = print_interval
    def __enter__(self):
        self.start_time = time.perf_counter_ns() 

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter_ns()
        execution_time = (self.end_time - self.start_time) / 1e6
        self.total_time += execution_time
        self.count += 1
        if self.count % self.print_interval == 0:
            print(f"#INFO: {self.context}: average execution time after {self.count} steps: {self.total_time / self.count:.9f} ms")


class ModuleOnDevice:
    def __init__(self, module, device):
        self.module = module
        self.prev_device = next(module.parameters()).device.type
        self.new_device = device
        self.different_device = self.prev_device != self.new_device

    def __enter__(self):
        if self.different_device:
            self.module.to(self.new_device)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.different_device:
            self.module.to(self.prev_device)


def get_args_from_json(json_file_path, args_dict):
    import json

    summary_filename = json_file_path
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)

    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]

    return args_dict


def mp4togif(path):
    try:
        import moviepy.editor as mp
    except:
        print("If you want to convert mp4 to gif, install package `moviepy`")
        return None

    if os.path.exists(path):
        clip = mp.VideoFileClip(path)
        if path.endswith(".mp4"):
            out_path = path[:-4] + ".gif"
        else:
            out_path = path + ".gif"
        clip.write_gif(out_path)
    else:
        print(f"`{path}` dose not exist")
