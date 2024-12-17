#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Recurrent Neural Network (RNN)
#  Update: 2021-03-05, Wenjun Zou: create RNN function


__all__ = [
    "DetermPolicy",
    "FiniteHorizonPolicy",
    "StochaPolicy",
    "ActionValue",
    "ActionValueDis",
    "StateValue",
]


import numpy as np
import torch
import torch.nn as nn
from torch.nn import RNN, LSTM, GRU
from gops.utils.common_utils import get_activation_func
from gops.utils.act_distribution_cls import Action_Distribution

# Define MLP function
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class DetermPolicy(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"][1]
        act_dim = kwargs["act_dim"]
        action_high_limit = kwargs["act_high_lim"]
        action_low_limit = kwargs["act_low_lim"]
        hidden_sizes = kwargs["hidden_sizes"]
        rnn_type = kwargs["rnn_type"]
        pi_sizes = list(hidden_sizes) + [act_dim]
        rnn_type_dict = {'RNN': RNN, 'LSTM' : LSTM, 'GRU': GRU}


        input_dim = obs_dim
        hidden_size = hidden_sizes[0]
        num_layers = kwargs["num_layers"]
        # Construct RNN
        self.rnn = rnn_type_dict[rnn_type](input_dim, hidden_size, num_layers, batch_first=True)
        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.register_buffer("act_high_lim", torch.from_numpy(action_high_limit))
        self.register_buffer("act_low_lim", torch.from_numpy(action_low_limit))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        output, _ = self.rnn(obs)
        output = output[:, -1, :]
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
            self.pi(output)
        ) + (self.act_high_lim + self.act_low_lim) / 2
        return action


class FiniteHorizonPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy for finite-horizon.
    Input: observation, time step.
    Output: action.
    """

    def __init__(self, **kwargs):
        raise NotImplementedError


class StochaPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of stochastic policy.
    Input: observation.
    Output: parameters of action distribution.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        rnn_type = kwargs["rnn_type"]
        num_layers = kwargs["num_layers"]
        action_high_limit = kwargs["act_high_lim"]
        action_low_limit = kwargs["act_low_lim"]
        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.std_sype = kwargs["std_type"]
        self.action_distribution_cls = kwargs["action_distribution_cls"]

        pi_sizes = list(hidden_sizes) + [act_dim]
        
        input_dim = obs_dim
        hidden_size = hidden_sizes[0]
        rnn_type_dict = {'RNN': RNN, 'LSTM' : LSTM, 'GRU': GRU}
        # Construct RNN
        self.rnn = rnn_type_dict[rnn_type](input_dim, hidden_size, num_layers, batch_first=True)
        self.mean = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        if self.std_sype == "parameter":
            self.log_std = nn.Parameter(
                torch.zeros(1,act_dim, dtype=torch.float32), requires_grad=True
            )
        else:
            self.log_std = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
        self.register_buffer("act_high_lim", torch.from_numpy(action_high_limit))
        self.register_buffer("act_low_lim", torch.from_numpy(action_low_limit))

    def forward(self, obs):
        output, _ = self.rnn(obs)
        output = output[:, -1, :]
        action_mean = self.mean(output)
        if self.std_sype == "parameter":
            action_log_std = self.log_std + torch.zeros_like(action_mean)
            action_std = torch.clamp(
                action_log_std, self.min_log_std, self.max_log_std
            ).exp()
        else:
            action_std = torch.clamp(
                self.log_std(output), self.min_log_std, self.max_log_std
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
        obs_dim = kwargs["obs_dim"][1]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        rnn_type = kwargs["rnn_type"]
        num_layers = kwargs["num_layers"]

        input_dim = obs_dim
        hidden_size = hidden_sizes[0]
        rnn_type_dict = {'RNN': RNN, 'LSTM' : LSTM, 'GRU': GRU}
        # Construct RNN
        self.rnn = rnn_type_dict[rnn_type](input_dim, hidden_size, num_layers, batch_first=True)
        self.q = mlp(
            list([hidden_sizes[0] + act_dim]) + list(hidden_sizes[1:]) + [1],
            get_activation_func(kwargs["hidden_activation"]),
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs, act):
        output, _ = self.rnn(obs)
        output = output[:, -1, :]
        input = torch.cat([output, act], dim=-1)
        q = self.q(input)
        return torch.squeeze(q, -1)


class ActionValueDis(nn.Module, Action_Distribution):
    """
    Approximated function of action-value function for discrete action space.
    Input: observation.
    Output: action-value for all action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"][1]
        act_num = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        rnn_type = kwargs["rnn_type"]
        num_layers = kwargs["num_layers"]

        input_dim = obs_dim
        hidden_size = hidden_sizes[0]
        rnn_type_dict = {'RNN': RNN, 'LSTM' : LSTM, 'GRU': GRU}
        # Construct RNN
        self.rnn = rnn_type_dict[rnn_type](input_dim, hidden_size, num_layers, batch_first=True)
        self.q = mlp(
            list(hidden_sizes) + [act_num],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs, act):
        output, _ = self.rnn(obs)
        output = output[:, -1, :]
        return self.q(torch.cat([output,act], dim =-1))


class StateValue(nn.Module, Action_Distribution):
    """
    Approximated function of state-value function.
    Input: observation, action.
    Output: state-value.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        rnn_type = kwargs["rnn_type"]
        num_layers = kwargs["num_layers"]

        input_dim = obs_dim
        hidden_size = hidden_sizes[0]
        rnn_type_dict = {'RNN': RNN, 'LSTM' : LSTM, 'GRU': GRU}
        # Construct RNN
        self.rnn = rnn_type_dict[rnn_type](input_dim, hidden_size, num_layers, batch_first=True)
        self.v = mlp(
            list(hidden_sizes) + [1],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        output, _ = self.rnn(obs)
        output = output[:, -1, :]
        v = self.v(output)
        return torch.squeeze(v, -1)

class ActionValueDistri(nn.Module):
    """
    Approximated function of distributed action-value function.
    Input: observation.
    Output: parameters of action-value distribution.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"][1]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        rnn_type = kwargs["rnn_type"]
        num_layers = kwargs["num_layers"]

        input_dim = obs_dim
        hidden_size = hidden_sizes[0]
        rnn_type_dict = {'RNN': RNN, 'LSTM' : LSTM, 'GRU': GRU}
        self.rnn = rnn_type_dict[rnn_type](input_dim, hidden_size, num_layers, batch_first=True)
        self.q = mlp(
           list([hidden_sizes[0] + act_dim]) + list(hidden_sizes[1:]) + [2],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.denominator = max(abs(self.min_log_std), self.max_log_std)

    def forward(self, obs, act):
        output, _ = self.rnn(obs)
        output = output[:, -1, :]
        input = torch.cat([output, act], dim=-1)
        logits = self.q(input)

        value_mean, log_std = torch.chunk(logits, chunks=2, dim=-1)

        value_log_std = torch.clamp_min(
            self.max_log_std * torch.tanh(log_std / self.denominator), 0
        ) + torch.clamp_max(
            -self.min_log_std * torch.tanh(log_std / self.denominator), 0
        )
        return torch.cat((value_mean, value_log_std), dim=-1)