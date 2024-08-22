#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Attention network
#  Update: 2023-07-03, Tong Liu: create attention
__all__ = [
    "PINet",
    "DetermPolicy",
    "FiniteHorizonPolicy",
    "FiniteHorizonFullPolicy",
    "StochaPolicy",
    "ActionValue",
    "ActionValueDis",
    "ActionValueDistri",
    "ActionValueDistriMultiR",
]
import numpy as np
import torch
import warnings
import itertools
import torch.nn as nn
from typing import Tuple
from functools import reduce
from abc import abstractmethod, ABCMeta
from gops.utils.act_distribution_cls import Action_Distribution
from gops.utils.common_utils import get_activation_func, FreezeParameters
from gops.apprfunc.mlp import mlp



def init_weights(m):
    if isinstance(m, nn.Linear):
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()


class MultiHeadProj(nn.Module):
    def __init__(self, d_model, head_num, head_dim, obj_num):
        super().__init__()
        self.d_model = d_model
        self.head_num = head_num
        self.head_dim = head_dim
        self.obj_num = obj_num
        self.projs = nn.ModuleList(
            [nn.Linear(head_dim, head_dim) for _ in range(head_num)])
        
    def forward(self, embeddings):
        # embeddings: [B, N, d_model]
        embeddings = embeddings.reshape(-1, self.obj_num, self.head_num, self.head_dim) # [B, N, head_num, head_dim]
        proj_value = [proj(embeddings[:,:, i]) for i, proj in enumerate(self.projs)]
        proj_value = torch.stack(proj_value, dim=2) # [B, N, head_num, head_dim]
        return proj_value

class PINet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.begin = kwargs["pi_begin"]
        self.end = kwargs["pi_end"]
        self.d_encodings = self.end - self.begin
        self.enable_mask = kwargs["enable_mask"]
        self.d_obj = kwargs["obj_dim"]
        self.enable_self_attention = kwargs.get("enable_self_attention", False)
        self.use_multi_head = kwargs.get("use_multi_head", False) # for backward compatibility
        assert self.d_encodings % self.d_obj == 0
        self.num_objs = int(self.d_encodings / self.d_obj)
        if self.enable_mask:
            self.pi_in_dim = self.d_obj -1 # the last dimension is mask
        else:
            self.pi_in_dim = self.d_obj
        self.pi_out_dim = kwargs.get("pi_out_dim", self.pi_in_dim*self.num_objs +1)


        self.encoding_others = kwargs["encoding_others"]
        obs_dim = kwargs["obs_dim"]
        self.others_in_dim = obs_dim - self.d_encodings
        if self.encoding_others:
            self.others_out_dim = kwargs["others_out_dim"]
        else: 
            self.others_out_dim = self.others_in_dim
        self.output_dim = self.others_out_dim + self.pi_out_dim
        hidden_sizes = kwargs["pi_hidden_sizes"]
        pi_sizes =  [self.pi_in_dim] + list(hidden_sizes) + [self.pi_out_dim]
        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["pi_hidden_activation"]),
            get_activation_func(kwargs["pi_output_activation"]),
        )
        init_weights(self.pi)
        if self.encoding_others:
            warnings.warn("encoding_others is enabled")
            self.others_encoder = mlp(
                [self.others_in_dim] + list(kwargs["others_hidden_sizes"]) + [self.others_out_dim],
                get_activation_func(kwargs["others_hidden_activation"]),
                get_activation_func(kwargs["others_output_activation"]),
            )
            init_weights(self.others_encoder)
        else:
            self.others_encoder = nn.Identity()

        if self.enable_self_attention:
            embedding_dim = self.pi_out_dim
            self.embedding_dim = embedding_dim
            warnings.warn("self_attention is enabled")
            if kwargs.get("attn_dim") is not None:
                self.attn_dim = kwargs["attn_dim"]
            else:
                self.attn_dim = self.pi_out_dim # default attn_dim is pi_out_dim
                warnings.warn("attn_dim is not specified, using pi_out_dim as attn_dim")
                if kwargs.get("head_num") is None:
                    head_num = 1
                    warnings.warn("head_num is not specified, using 1 as head_num")
                else:
                    head_num = kwargs["head_num"]
                head_dim = embedding_dim // head_num
                self.head_dim = head_dim
                self.head_num = head_num
                self.Wq =  nn.Linear(self.others_out_dim, head_dim*head_num)
                if self.use_multi_head:
                    self.Wk =  MultiHeadProj(embedding_dim, head_num, head_dim, self.num_objs)
                    self.Wv =  MultiHeadProj(embedding_dim, head_num, head_dim, self.num_objs)
                else:   
                    self.Wk =  nn.Linear(head_dim, head_dim) # BUG: only for backward compatibility the code is wrong
                    self.Wv =  nn.Linear(head_dim, head_dim)
            self.attn_weights = None 

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        objs = obs[:, self.begin:self.end]
        others = torch.cat([obs[:, :self.begin], obs[:, self.end:]], dim=1)
        objs = objs.reshape(-1, self.num_objs, self.d_obj)
        others = self.others_encoder(others)

        if self.enable_mask:
            mask = objs[:, :, -1]
            objs = objs[:, :, :-1]
        else:
            mask = torch.ones_like(objs[:, :, 0]) # [B, N]
        
        
        embeddings = self.pi(objs)*mask.unsqueeze(-1) # [B, N, d_model]

        if self.enable_self_attention:
            query = self.Wq(others).reshape(-1,self.head_num,1, self.head_dim) # [B, head_num, 1, head_dim]
            if self.use_multi_head:
                key = self.Wk(embeddings) # [B, N, head_num, head_dim]
                value = self.Wv(embeddings) # [B, N, head_num, head_dim]
            else:
                reshaped_embeddings = embeddings.reshape(-1, self.num_objs, self.head_num, self.head_dim) # [B, N, head_num, head_dim]
                key = self.Wk(reshaped_embeddings) # [B, N, head_num, head_dim]
                value = self.Wv(reshaped_embeddings) # [B, N, head_num, head_dim]
            value = value*mask.unsqueeze(-1).unsqueeze(-1) # [B, N, head_num, head_dim]
            # logits = torch.einsum("nqhd,nkhd->nhqk", [query, key]) / np.sqrt(self.embedding_dim) # [B, head_num, 1, N] donot use einsum
            key = key.permute(0, 2, 1, 3)  # 形状变为  [B, head_num, N, head_dim]

            # 进行矩阵乘法
            logits = torch.matmul(query, key.transpose(-2, -1))  # 形状变为 [B, head_num, 1, N]

            # 缩放
            logits = logits / np.sqrt(self.embedding_dim)  # head_dim 即 self.embedding_dim
            logits = logits.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
            attn_weights = torch.softmax(logits, axis=-1) # [B, head_num, 1, N]
            # embeddings = torch.einsum("nhqk,nkhd->nqhd", [attn_weights, value]).reshape(-1, self.embedding_dim) # [B, d_model]
            attn_p = attn_weights
            value_p = value.permute(0, 2, 1, 3) # [B, head_num, N, head_dim]
            embeddings = torch.matmul(attn_p, value_p).reshape(-1, self.embedding_dim) # [B, d_model]
            self.attn_weights = attn_weights.squeeze(2).sum(1)/self.head_num



            
            # query = embeddings.sum(axis=-2) / (mask.sum(axis=-1) + 1e-5).unsqueeze(axis=-1) # [b, d_model] / [B, 1] --> [B, d_model]
            # query = torch.concat([query, others], dim=1) # [B, d_model + d_others]
            # logits = torch.bmm(self.Uq(query).unsqueeze(1), self.Ur(embeddings).transpose(-1, -2)).squeeze(1) / np.sqrt(self.attn_dim) # [B, N]
            # logits = logits + ((1 - mask) * -1e9)  # mask ==1 means the object is the true vehicle
            # self.attn_weights = torch.softmax(logits, axis=-1) # [B, N]
            # # self.attn_weights = (self.attn_weights + 0*mask)
            # # self.attn_weights = self.attn_weights / (self.attn_weights.sum(axis=-1, keepdim=True) + 1e-5)
            # # print(self.attn_weights)
            # embeddings = torch.matmul(self.attn_weights.unsqueeze(axis=1),embeddings).squeeze(axis=-2) # [B, d_model]
        else:
            embeddings = embeddings.sum(dim=1, keepdim=False) # [B, d_model]
        
        return torch.cat([others, embeddings], dim=1)
    


class DetermPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy.
    Input: observation.
    Output: action.
    """

    def __init__(self, **kwargs):
        raise NotImplementedError


class FiniteHorizonPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy for finite-horizon.
    Input: observation, time step.
    Output: action.
    """

    def __init__(self, **kwargs):
        raise NotImplementedError


class FiniteHorizonFullPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy for finite-horizon.
    Input: observation, time step.
    Output: action.
    """

    def __init__(self, **kwargs):
        raise NotImplementedError


# Stochastic Policy
class StochaPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of stochastic policy.
    Input: observation.
    Output: parameters of action distribution.
    """

    def __init__(self, **kwargs):
        super().__init__()
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.std_type = kwargs["std_type"]
        self.pi_net = kwargs["pi_net"]
        self.freeze_pi_net = kwargs["freeze_pi_net"] == "actor" or kwargs["freeze_pi_net"] == "both"
        input_dim = self.pi_net.output_dim

        # mean and log_std are calculated by different MLP
        if self.std_type == "mlp_separated":
            pi_sizes = [input_dim] + list(hidden_sizes) + [act_dim]
            self.mean = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            self.log_std = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
        # mean and log_std are calculated by same MLP
        elif self.std_type == "mlp_shared":
            pi_sizes = [input_dim] + list(hidden_sizes) + [act_dim * 2]
            self.policy = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
        # mean is calculated by MLP, and log_std is learnable parameter
        elif self.std_type == "parameter":
            pi_sizes = [input_dim] + list(hidden_sizes) + [act_dim]
            self.mean = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            self.log_std = nn.Parameter(-0.5*torch.ones(1, act_dim))

        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def shared_paras(self):
        return self.pi_net.parameters()

    def ego_paras(self):
        return itertools.chain(*[modules.parameters() for modules in self.children() if modules != self.pi_net])
        

    def forward(self, obs):
        with FreezeParameters([self.pi_net], self.freeze_pi_net):
            encoding = self.pi_net(obs)
        if self.std_type == "mlp_separated":
            action_mean = self.mean(encoding)
            action_std = torch.clamp(
                self.log_std(encoding), self.min_log_std, self.max_log_std
            ).exp()
        elif self.std_type == "mlp_shared":
            logits = self.policy(encoding)
            action_mean, action_log_std = torch.chunk(
                logits, chunks=2, dim=-1
            )  # output the mean
            action_std = torch.clamp(
                action_log_std, self.min_log_std, self.max_log_std
            ).exp()
        elif self.std_type == "parameter":
            action_mean = self.mean(encoding)
            action_log_std = self.log_std + torch.zeros_like(action_mean)
            action_std = torch.clamp(
                action_log_std, self.min_log_std, self.max_log_std
            ).exp()

        return torch.cat((action_mean, action_std), dim=-1)


class ActionValue(nn.Module, Action_Distribution):
    """
    Approximated function of action-value function.
    Input: observation, action.
    Output: action-value.
    """

    def __init__(self, **kwargs):
        super().__init__()
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.pi_net = kwargs["pi_net"]
        self.freeze_pi_net = kwargs["freeze_pi_net"] == "critic" or kwargs["freeze_pi_net"] == "both"
        input_dim = self.pi_net.output_dim + act_dim

        self.q = mlp(
            [input_dim] + list(hidden_sizes) + [1],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]
        rew_comp_dim = kwargs["additional_info"]["reward_comps"]["shape"][0]
        self.rew_pred_head = mlp(
            [input_dim] +[64]+ [rew_comp_dim],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )

    def shared_paras(self):
        return self.pi_net.parameters()
    
    def ego_paras(self):
        return itertools.chain(*[modules.parameters() for modules in self.children() if modules != self.pi_net])

    def forward(self, obs, act):
        with FreezeParameters([self.pi_net], self.freeze_pi_net):
            encoding = self.pi_net(obs)
        q = self.q(torch.cat([encoding, act], dim=-1))
        return torch.squeeze(q, -1)
    
    def predict_reward(self, obs, act):
        encoding = self.pi_net(obs)
        return self.rew_pred_head(torch.cat([encoding, act], dim=-1))

class ActionValueDis(nn.Module, Action_Distribution):
    """
    Approximated function of action-value function for discrete action space.
    Input: observation.
    Output: action-value for all action.
    """

    def __init__(self, **kwargs):
        raise NotImplementedError

class ActionValueDistri(nn.Module):
    """
    Approximated function of distributed action-value function.
    Input: observation.
    Output: parameters of action-value distribution.
    """

    def __init__(self, **kwargs):
        super().__init__()
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.pi_net = kwargs["pi_net"]
        self.freeze_pi_net = kwargs["freeze_pi_net"] == "critic"
        self.std_type = kwargs["std_type"]
        input_dim = self.pi_net.output_dim + act_dim
        if self.std_type == "mlp_shared":
            self.q = mlp(
                [input_dim] + list(hidden_sizes) + [2],
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
        elif self.std_type == "mlp_separated":
            self.q = mlp(
                [input_dim] + list(hidden_sizes) + [1],
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            self.q_std = mlp(
                [input_dim] + list(hidden_sizes) + [1],
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
        else:
            raise NotImplementedError
        if "min_log_std"  in kwargs or "max_log_std" in kwargs:
            warnings.warn("min_log_std and max_log_std are deprecated in ActionValueDistri.")
        if kwargs.get("additional_info") is None:
            pass
        elif kwargs["additional_info"].get("reward_comps") is None:
            pass
        else:        
            rew_comp_dim = kwargs["additional_info"]["reward_comps"]["shape"][0]
            self.rew_pred_head = mlp(
                [input_dim] +[64]+ [rew_comp_dim],
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )

    def shared_paras(self):
        return self.pi_net.parameters()

    def ego_paras(self):
        return itertools.chain(*[modules.parameters() for modules in self.children() if modules != self.pi_net])

    def forward(self, obs, act):
        
        with FreezeParameters([self.pi_net], self.freeze_pi_net):
            encoding = self.pi_net(obs)
        if self.std_type == "mlp_shared":
            logits = self.q(torch.cat([encoding, act], dim=-1))
            value_mean, value_std = torch.chunk(logits, chunks=2, dim=-1)
            value_std = torch.nn.functional.softplus(value_std) 

        elif self.std_type == "mlp_separated":
            value_mean = self.q(torch.cat([encoding, act], dim=-1))
            value_std = torch.nn.functional.softplus(self.q_std(torch.cat([encoding, act], dim=-1)))

        return torch.cat((value_mean, value_std), dim=-1)
    
    def predict_reward(self, obs, act):
        encoding = self.pi_net(obs)
        return self.rew_pred_head(torch.cat([encoding, act], dim=-1))
    


class ActionValueDistriMultiR(nn.Module):
    """
    Approximated function of distributed action-value function.
    Input: observation.
    Output: parameters of action-value distribution.
    """

    def __init__(self, **kwargs):
        super().__init__()
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.pi_net = kwargs["pi_net"]
        self.freeze_pi_net = kwargs["freeze_pi_net"] == "critic"
        input_dim = self.pi_net.output_dim + act_dim
        self.q = mlp(
            [input_dim] + list(hidden_sizes) + [2],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        if "min_log_std"  in kwargs or "max_log_std" in kwargs:
            warnings.warn("min_log_std and max_log_std are deprecated in ActionValueDistri.")

        #rew_comp_dim = reduce(lambda x, y: x + y, [value["shape"] for value in kwargs["additional_info"].values()])
        rew_comp_dim = kwargs["additional_info"]["reward_comps"]["shape"][0]
        self.q_comp = mlp(
            [input_dim] + list(hidden_sizes) + [rew_comp_dim],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )

    def shared_paras(self):
        return self.pi_net.parameters()

    def ego_paras(self):
        return itertools.chain(*[modules.parameters() for modules in self.children() if modules != self.pi_net])

    def forward(self, obs, act):
        
        with FreezeParameters([self.pi_net], self.freeze_pi_net):
            encoding = self.pi_net(obs)
        
        logits = self.q(torch.cat([encoding, act], dim=-1))
        value_mean, value_std = torch.chunk(logits, chunks=2, dim=-1)
        value_log_std = torch.nn.functional.softplus(value_std) 

        return torch.cat((value_mean, value_log_std), dim=-1)
    
    def cal_comp(self, obs, act):
        with FreezeParameters([self.pi_net], True):
            encoding = self.pi_net(obs)
        return self.q_comp(torch.cat([encoding, act], dim=-1))