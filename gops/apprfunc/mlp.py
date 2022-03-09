#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Multilayer Perceptron (MLP)
#  Update: 2021-03-05, Wenjun Zou: create MLP function


__all__ = ['DetermPolicy', 'StochaPolicy', 'ActionValue', 'ActionValueDis', 'ActionValueDistri', 'StateValue']

import numpy as np  # Matrix computation library
import torch
import torch.nn as nn
from gops.utils.utils import get_activation_func
from act_distribution_cls import Action_Distribution


# Define MLP function
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


# Count parameter number of MLP
def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


# Deterministic policy
class DetermPolicy(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']

        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes,
                      get_activation_func(kwargs['hidden_activation']),
                      get_activation_func(kwargs['output_activation']))
        self.register_buffer('act_high_lim', torch.from_numpy(kwargs['act_high_lim']))
        self.register_buffer('act_low_lim', torch.from_numpy(kwargs['act_low_lim']))
        self.action_distirbution_cls = kwargs['action_distirbution_cls']

    def forward(self, obs):
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(self.pi(obs)) \
                 + (self.act_high_lim + self.act_low_lim) / 2
        return action


# Stochastic Policy
class StochaPolicy(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']

        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.mean = mlp(pi_sizes,
                        get_activation_func(kwargs['hidden_activation']),
                        get_activation_func(kwargs['output_activation']))
        self.log_std = mlp(pi_sizes,
                           get_activation_func(kwargs['hidden_activation']),
                           get_activation_func(kwargs['output_activation']))
        self.min_log_std = kwargs['min_log_std']
        self.max_log_std = kwargs['max_log_std']
        self.register_buffer('act_high_lim', torch.from_numpy(kwargs['act_high_lim']))
        self.register_buffer('act_low_lim', torch.from_numpy(kwargs['act_low_lim']))
        self.action_distirbution_cls = kwargs['action_distirbution_cls']

    def forward(self, obs):
        action_mean = self.mean(obs)
        action_std = torch.clamp(self.log_std(obs), self.min_log_std, self.max_log_std).exp()
        return torch.cat((action_mean, action_std), dim=-1)


class ActionValue(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1],
                     get_activation_func(kwargs['hidden_activation']),
                     get_activation_func(kwargs['output_activation']))
        self.action_distirbution_cls = kwargs['action_distirbution_cls']

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class ActionValueDis(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_num = kwargs['act_num']
        hidden_sizes = kwargs['hidden_sizes']
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_num],
                     get_activation_func(kwargs['hidden_activation']),
                     get_activation_func(kwargs['output_activation']))
        self.action_distirbution_cls = kwargs['action_distirbution_cls']

    def forward(self, obs):
        return self.q(obs)


class ActionValueDistri(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        self.mean = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1],
                     get_activation_func(kwargs['hidden_activation']),
                     get_activation_func(kwargs['output_activation']))
        self.min_log_std = kwargs['min_log_std']
        self.max_log_std = kwargs['max_log_std']
        self.denominator = max(abs(self.min_log_std), self.max_log_std)

        self.log_std = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1],
                           get_activation_func(kwargs['hidden_activation']),
                           get_activation_func(kwargs['output_activation']))

    def forward(self, obs, act, min=False):
        value_mean = self.mean(torch.cat([obs, act], dim=-1))
        log_std = self.log_std(torch.cat([obs, act], dim=-1))

        value_log_std = torch.clamp_min(self.max_log_std * torch.tanh(log_std / self.denominator), 0) + \
                        torch.clamp_max(-self.min_log_std * torch.tanh(log_std / self.denominator), 0)
        return torch.cat((value_mean, value_log_std), dim=-1)


class StochaPolicyDis(ActionValueDis, Action_Distribution):
    pass


class StateValue(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        hidden_sizes = kwargs['hidden_sizes']
        self.v = mlp([obs_dim] + list(hidden_sizes) + [1],
                     get_activation_func(kwargs['hidden_activation']),
                     get_activation_func(kwargs['output_activation']))
        self.action_distirbution_cls = kwargs['action_distirbution_cls']

    def forward(self, obs):
        v = self.v(obs)
        return torch.squeeze(v, -1)

#
