#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Transform pkl network to onnx version
#  Update: 2023-01-05, Jiaxin Gao: Create codes

import contextlib
import torch, torch.nn as nn
import onnxruntime as ort
import argparse
import os
import sys
from gops.utils.common_utils import get_args_from_json
import numpy as np

py_file_path = os.path.abspath(__file__)
utils_path = os.path.dirname(py_file_path)
gops_path = os.path.dirname(utils_path)
# Add algorithm file to sys path
alg_file = "algorithm"
alg_path = os.path.join(gops_path, alg_file)
sys.path.append(alg_path)


def __load_args(log_policy_dir):
    log_policy_dir = log_policy_dir
    json_path = os.path.join(log_policy_dir, "config.json")
    parser = argparse.ArgumentParser()
    args_dict = vars(parser.parse_args())
    args = get_args_from_json(json_path, args_dict)
    return args

def export_model(model: nn.Module, example_obs: torch.Tensor, path: str):
    with _module_inference(model):
        inference_helper = _InferenceHelper(model)
        torch.onnx.export(inference_helper, example_obs, path, input_names=['input'], output_names=['output'],
                          opset_version=11)


@contextlib.contextmanager
def _module_inference(module: nn.Module):
    training = module.training
    module.train(False)
    yield
    module.train(training)


class _InferenceHelper(nn.Module):
    def __init__(self, model):
        super().__init__()

        from gops.apprfunc.mlp import Action_Distribution

        assert isinstance(model, nn.Module) and isinstance(
            model, Action_Distribution
        ), (
            "The model must inherit from nn.Module and Action_Distribution. "
            f"Got {model.__class__.__mro__}"
        )
        self.model = model

    def forward(self, obs: torch.Tensor):
        obs = obs.unsqueeze(0)
        logits = self.model(obs)
        act_dist = self.model.get_act_dist(logits)
        mode = act_dist.mode()
        return mode.squeeze(0)

class _InferenceHelper_FHADP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, obs: torch.Tensor):
        return self.model(obs, torch.ones(1))

def deterministic_policy_export_onnx_model(networks, input_dim, policy_dir):

    example = torch.rand(1, input_dim)  # network input dim
    output_onnx_model = policy_dir
    torch.onnx.export(networks.policy, example, output_onnx_model, input_names=['input'],
                      output_names=['output'], opset_version=11)
    
def deterministic_value_export_onnx_model(networks, input_dim, policy_dir):

    example = torch.rand(1, input_dim)  # network input dim
    output_onnx_model = policy_dir
    torch.onnx.export(networks.v, example, output_onnx_model, input_names=['input'],
                      output_names=['output'], opset_version=11)
    
def fhadp_policy_export_onnx_model(networks, input_dim, policy_dir):

    example = torch.rand(1, input_dim)  # network input dim
    output_onnx_model = policy_dir
    model = _InferenceHelper_FHADP(networks.policy)
    torch.onnx.export(model, example, output_onnx_model, input_names=['input', "input1"],
                      output_names=['output'], opset_version=11)

def deterministic_stochastic_export_onnx_model(networks, input_dim, policy_dir):

    example = torch.rand(1, input_dim)  # network input dim
    output_onnx_model = policy_dir
    model = networks.policy
    export_model(model, example, output_onnx_model)



if __name__=='__main__':

    # Load trained policy
    log_policy_dir = "../../results/pyth_idsim/FHADP2_231209-162345-only-tracking"
    args = __load_args(log_policy_dir)
    alg_name = args["algorithm"]
    print('alg_name = ', alg_name)
    alg_file_name = alg_name.lower()
    file = __import__(alg_file_name)
    ApproxContainer = getattr(file, "ApproxContainer")
    networks = ApproxContainer(**args)

    # Load trained policy
    log_path = log_policy_dir + "/apprfunc/apprfunc_{}.pkl".format(40000)  # network position
    networks.load_state_dict(torch.load(log_path))
    networks.eval()

    # create onnx model
    ### example of deterministic policy FHADP algorithm
    policy_input_dim = 202
    value_input_dim = 202
    policy_dir = '../../transform_onnx_network/idsim_policy.onnx'
    value_dir = '../../transform_onnx_network/idsim_value.onnx'
    deterministic_policy_export_onnx_model(networks, policy_input_dim, policy_dir)
    deterministic_value_export_onnx_model(networks, value_input_dim, value_dir)


    # ### example of stochastic policy sac algorithm
    # input_dim = 50
    # policy_dir = '../../transform_onnx_network/network_sac_ziqing.onnx'
    # deterministic_stochastic_export_onnx_model(networks, input_dim, policy_dir)

    # load onnx model for test
    ### example of deterministic policy FHADP algorithm
    ort_session_policy = ort.InferenceSession("../../transform_onnx_network/idsim_policy.onnx")
    example_policy = np.random.randn(1, 202).astype(np.float32)
    inputs_policy = {ort_session_policy.get_inputs()[0].name: example_policy}
    outputs_policy = ort_session_policy.run(None, inputs_policy)
    print(outputs_policy[0])
    action = networks.policy(torch.tensor(example_policy))
    print(action)

    ort_session_value = ort.InferenceSession("../../transform_onnx_network/idsim_value.onnx")
    example_value = np.random.randn(1, 202).astype(np.float32)
    inputs_value = {ort_session_value.get_inputs()[0].name: example_value}
    outputs_value = ort_session_value.run(None, inputs_value)
    print(outputs_value[0])
    value = networks.v(torch.tensor(example_value))
    print(value)

    # ### example of stochastic policy sac algorithm
    # ort_session = ort.InferenceSession("../../transform_onnx_network/network_sac_ziqing.onnx")
    # example1 = np.random.randn(1, 50).astype(np.float32)
    # inputs = {ort_session.get_inputs()[0].name: example1}
    # outputs = ort_session.run(None, inputs)
    # print(outputs)
    # action = networks.policy(torch.tensor(example1))
    # act_dist = model.get_act_dist(action).mode()
    # print(act_dist)