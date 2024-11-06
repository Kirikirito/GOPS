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

import os
import sys
import argparse
import importlib
import numpy as np
import torch
import time
import json

import torch.nn as nn
import onnxruntime as ort

from pathlib import Path
from typing import Callable, Dict, Union, Optional, Any
from gops.utils.common_utils import get_args_from_json



class PolicyExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, obs_act_info: Dict[str, Any]):
        super().__init__()
        self.model = model
        self.obs_act_info = obs_act_info

    def forward(self, obs: torch.Tensor):
        obs_scale_factor = torch.tensor(self.obs_act_info["obs_scale_factor"]).float()
        act_scale_factor = torch.tensor(self.obs_act_info["act_scale_factor"]).float()
        bias = torch.tensor(self.obs_act_info["act_scale_bias"]).float()
        obs = obs*obs_scale_factor
        logits = self.model.policy(obs)
        action_distribution = self.model.create_action_distributions(logits)
        action = action_distribution.mode().float()
        real_act = action*act_scale_factor + bias
        return real_act

class QExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, obs_act_info: Dict[str, Any]):
        super().__init__()
        if hasattr(model, "q1"):
            self.double_q = True
        elif hasattr(model, "q"):
            self.double_q = False
        self.model = model 
        self.obs_act_info = obs_act_info

    def forward(self, obs_act: torch.Tensor):
        act_dim = self.obs_act_info["act_dim"]
        obs_scale_factor = torch.tensor(self.obs_act_info["obs_scale_factor"]).float()
        obs  = obs_act[...,:-act_dim] 
        obs = obs*obs_scale_factor
        act = obs_act[...,-act_dim:]
        if self.double_q:
            logits_1 = self.model.q1(obs,act)
            logits_2 = self.model.q2(obs,act)
            q_value_1 = logits_1[...,0:1]
            q_value_2 = logits_2[...,0:1]
            q_value = torch.min(q_value_1, q_value_2)
        else:
            logits = self.model.q(obs,act)
            q_value = logits[...,0:1]
        return q_value



class OnnxExporter:
    def __init__(self, 
                 model_path: str,
                 checkpoint: Union[int, str],
                 export_name: str,
                 seq_len: int,
                 export_path: str = './'

    ):
        self.model_path = Path(model_path)
        self.checkpoint = checkpoint
        self.export_name = export_name
        self.seq_len = seq_len
        self.export_path = Path(export_path)
        time_str = time.strftime("%y%m%d_%H%M%S", time.localtime())
        self.export_path = self.export_path/time_str
        self.args = self._load_args(self.model_path)
        self.model = self._load_model()
        self.obs_act_info = self._get_obs_act_info()


    def _load_args(self, log_policy_dir):
        log_policy_dir = log_policy_dir
        json_path = os.path.join(log_policy_dir, "config.json")
        with open(json_path, 'r') as f:
            args = json.load(f)
        return args
    
    def _get_obs_act_info(self):
        obs_dim = self.args["obsv_dim"]
        act_dim = self.args["action_dim"]
        ego_dim = self.args["env_model_config"]["ego_feat_dim"]
        sur_dim = self.args["env_model_config"]["per_sur_feat_dim"] + 3 # +3 for length, width, mask
        ref_dim = self.args["env_model_config"]["per_ref_feat_dim"]
        num_ref_points = len(self.args["env_model_config"]["downsample_ref_point_index"])
        num_objs = int(sum(i for i in self.args["env_config"]["obs_num_surrounding_vehicles"].values()))
        obs_scale_factor = self.args["obs_scale"]
        act_scale_factor = 1
        act_scale_bias = 0
        obs_act_info = {
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "ego_dim": ego_dim,
            "sur_dim": sur_dim,
            "ref_dim": ref_dim,
            "num_objs": num_objs,
            "num_ref_points": num_ref_points,
            "obs_scale_factor": obs_scale_factor,
            "act_scale_factor": act_scale_factor,
            "act_scale_bias": act_scale_bias
        }
        return obs_act_info
    
    def _load_model(self):
        args = self._load_args(self.model_path)
        alg_name = args["algorithm"]
        alg_file_name = alg_name.lower()
        mdl = importlib.import_module("gops.algorithm." + alg_file_name)
        print(f"NOTE: Using the default path of gops in the current environment, not necessarily the path of the current file.")
        ApproxContainer = getattr(mdl, "ApproxContainer")
        networks = ApproxContainer(**args)
        self.model_path = self.model_path / "apprfunc"/f"apprfunc_{self.checkpoint}.pkl" # network position
        networks.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')), strict=False)
        networks.eval()
        return networks
    

    def export(self):
        self.export_policy()
        self.export_q()
        self.save_origin_info()


    def export_policy(self):
        model_with_wrapper = PolicyExportWrapper(self.model, self.obs_act_info)
        input_dim = self.obs_act_info["obs_dim"]
        if self.seq_len > 1:
            example_input = torch.rand(1, self.seq_len, input_dim)  # network input dim
        else:
            example_input = torch.rand(1, input_dim)
        self._export_onnx_model(model_with_wrapper, example_input, "policy")

    def export_q(self):
        model_with_wrapper = QExportWrapper(self.model, self.obs_act_info)
        input_dim = self.obs_act_info["obs_dim"] + self.obs_act_info["act_dim"]
        if self.seq_len > 1:
            example_input = torch.rand(1, self.seq_len, input_dim)
        else:
            example_input = torch.rand(1, input_dim)
        self._export_onnx_model(model_with_wrapper, example_input, "q")

    def save_origin_info(self):
        self.export_path.mkdir(parents=True, exist_ok=True)
        with open(self.export_path/'config.json', 'w') as f:
            json.dump(self.args, f, indent=4)
        # copy the original model file
        model_file = self.export_path/f"apprfunc_{self.checkpoint}.pkl"
        os.system(f"cp {self.model_path} {model_file}")
        print(f"Saved the original model file to {model_file}")
        

        
        

        


    def _export_onnx_model(self, model_with_wrapper, example_input, surfix):
        self.export_path.mkdir(parents=True, exist_ok=True)
        exported_onnx_model = self.export_path/('_'.join([self.export_name, surfix]) + '.onnx')
        torch.onnx.export(model_with_wrapper, example_input, exported_onnx_model, input_names=['input'], output_names=['output'], )
        print(f"Exported onnx model to {exported_onnx_model}")
        ort_session = ort.InferenceSession(exported_onnx_model)
        self._verify_onnx_model(ort_session, model_with_wrapper, example_input)
        

    def _verify_onnx_model(self, ort_session, model_with_wrapper, example):
        ort_inputs = {ort_session.get_inputs()[0].name: example.detach().numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        model_outputs = model_with_wrapper(example)
        assert (np.allclose(ort_outputs[0], model_outputs.detach().numpy(), rtol=1e-03, atol=1e-05)), "The onnx model is not consistent with the original model, please check the model."

     



    
def main():

    model_path = "/root/gops/results/idsim/idsim_multilane_exp_0907/idsim_multilane_vec/dsact_pi/12345_2000000_run0"
    checkpoint = 1000000
    export_name = "wwx_0907_ml_punish"
    seq_len = 1
    export_path = "./"
    parser = argparse.ArgumentParser(description='Export pkl network to onnx version')

    parser.add_argument('-mc', '--model_path', type=str, default=model_path, help='The path of the model')
    parser.add_argument('-cp', '--checkpoint', type=int, default=checkpoint, help='The checkpoint of the model')
    parser.add_argument('-en','--export_name', type=str, default=export_name, help='The name of the exported onnx model')
    parser.add_argument('-sl','--seq_len', type=int, default=seq_len, help='The sequence length of the input')
    parser.add_argument('-ep','--export_path', type=str, default=export_path, help='The path to save the exported onnx model')
    args = parser.parse_args()
    exporter = OnnxExporter(args.model_path, args.checkpoint, args.export_name, args.seq_len, args.export_path)
    exporter.export()

if __name__=='__main__':
    main()
