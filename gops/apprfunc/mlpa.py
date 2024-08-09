#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Multilayer Perceptron (MLP)
#  Update: 2021-03-05, Wenjun Zou: create MLP function
#  Update: 2023-07-28, Jiaxin Gao: add FiniteHorizonFullPolicy function
#  Update: 2023-10-25, Wenxuan Wang: add DSAC-T algorithm


__all__ = [
    "DetermPolicy",
    "FiniteHorizonPolicy",
    "FiniteHorizonFullPolicy",
    "MultiplierNet",
    "StochaPolicy",
    "ActionValue",
    "ActionValueDis",
    "ActionValueDistri",
    "StochaPolicyDis",
    "StateValue",
]

import numpy as np
import torch
import warnings
import torch.nn as nn
from gops.utils.common_utils import get_activation_func
from gops.utils.act_distribution_cls import Action_Distribution


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

class AMLP(nn.Module):
    def __init__(self, sizes, activation, output_activation=nn.Identity,adapter_layers=None, adapter_start_layer=0, add_bias=True, consider_last_act=False, act_dim = None):
        super().__init__()
        self.sizes = sizes
        self.activation = activation
        self.output_activation = output_activation
        self.adapter_layers = adapter_layers
        self.adapter_start_layer = adapter_start_layer
        self.add_bias = add_bias
        self.consider_last_act = consider_last_act
        self.act_dim = act_dim
        self._build_layers()
        self._init_weights()
        self._enable_adapter = False
    def _build_layers(self):
        if self.adapter_layers is None:
            layers = []
            for j in range(len(self.sizes) - 1):
                act = self.activation if j < len(self.sizes) - 2 else self.output_activation
                layers += [nn.Linear(self.sizes[j], self.sizes[j + 1], bias=self.add_bias), act()]
                self.main_route = nn.Sequential(*layers)

        else:
            assert self.adapter_start_layer < len(self.sizes) - 1 and self.adapter_start_layer >= 0 ,  "adapter_start_layer should be in [0, len(sizes)-1)"
            layers_main = []
            layers_adapter = []
            if self.consider_last_act:
                adapter_start_size = self.sizes[self.adapter_start_layer] + self.act_dim
            else:
                adapter_start_size = self.sizes[self.adapter_start_layer]
            self.adapter_layers= (adapter_start_size,*self.adapter_layers, self.sizes[-1]) # add output layer
            for j in range(len(self.sizes) - 1):
                act = self.activation if j < len(self.sizes) - 2 else self.output_activation
                layers_main += [nn.Linear(self.sizes[j], self.sizes[j + 1], bias=self.add_bias), act()]
            for  j in range(len(self.adapter_layers) - 1):
                act = self.activation if j < len(self.adapter_layers) - 2 else self.output_activation
                layers_adapter += [nn.Linear(self.adapter_layers[j], self.adapter_layers[j + 1], bias=self.add_bias), act()]
            self.main_route_head = nn.Sequential(*layers_main[:self.adapter_start_layer])
            self.adapter_route = nn.Sequential(*layers_adapter)
            self.main_route_tail = nn.Sequential(*layers_main[self.adapter_start_layer:])
            self.main_route = nn.Sequential(self.main_route_head, self.main_route_tail)

    def _init_weights(self):
        for m in self.main_route.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        if self.adapter_layers is not None:
            self._near_zero_init(self.adapter_route)
    def _near_zero_init(self, module):
        # the last layer of adapter should use small initialization gain to ensure the adapter is close to zero at the beginning
        linear_layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
        for layer in linear_layers:
            nn.init.kaiming_normal_(layer.weight)
            if not self.add_bias:
                nn.init.constant_(layer.bias, 0)
        last_layer = linear_layers[-1]
        nn.init.xavier_uniform_(last_layer.weight, gain=0.01)
        nn.init.constant_(last_layer.bias, 0)

        
    def forward(self, x, last_act):
        if  self._enable_adapter:
            if self.consider_last_act:
                is_vilid = last_act.norm(p=1, dim=-1) > 0 # if last_act is valid, only for 1D action 
                x_main = self.main_route_head(x)
                x_adapter = self.adapter_route(torch.cat([x_main, last_act], dim=-1))
                x_main = self.main_route_tail(x_main)
                return x_main, x_adapter*is_vilid.unsqueeze(-1)
            else:
                x_main = self.main_route_head(x)
                x_adapter = self.adapter_route(x_main)
                x_main = self.main_route_tail(x_main)
                return x_main, x_adapter
        else:
            return self.main_route(x)
        
    def enable_adapter(self):
        assert self.adapter_layers is not None, "adapter_layer is None, can not enable adapter"
        self._enable_adapter = True
        for p in self.main_route.parameters(): # freeze main route
            p.requires_grad = False




# Deterministic policy
class DetermPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy.
    Input: observation.
    Output: action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        adapter_layers = kwargs.get("adapter_layers", None)
        adapter_start_layer = kwargs.get("adapter_start_layer", 0)
        add_bias = kwargs.get("adapter_add_bias", True)

        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = AMLP(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
            adapter_layers=adapter_layers,
            adapter_start_layer=adapter_start_layer,
            add_bias=add_bias 
        )
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]
        self._enable_adapter = kwargs.get("enable_adapter", False)
        self.adaption_ratio = kwargs.get("adaption_ratio", 0.3)

    def forward(self, obs):
        if self._enable_adapter:
            main_action, adapter = self.pi(obs)
            action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(main_action) + \
            self.adaption_ratio*(self.act_high_lim + self.act_low_lim) / 2*torch.tanh(adapter)
            action = torch.clamp(action, self.act_low_lim, self.act_high_lim)
        else:
            action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
                self.pi(obs)
            ) + (self.act_high_lim + self.act_low_lim) / 2
        return action

    def enable_adapter(self):
        self.pi.enable_adapter()
        self._enable_adapter = True


class FiniteHorizonPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy for finite-horizon.
    Input: observation, time step.
    Output: action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"] + 1
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]

        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs, virtual_t=1):
        virtual_t = virtual_t * torch.ones(
            size=[obs.shape[0], 1], dtype=torch.float32, device=obs.device
        )
        expand_obs = torch.cat((obs, virtual_t), 1)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
            self.pi(expand_obs)
        ) + (self.act_high_lim + self.act_low_lim) / 2
        return action


class MultiplierNet(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy for finite-horizon.
    Input: observation, time step.
    Output: action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"] + 1
        hidden_sizes = kwargs["hidden_sizes"]

        pi_sizes = [obs_dim] + list(hidden_sizes) + [1]
        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )

    def forward(self, obs, virtual_t=1):
        virtual_t = virtual_t * torch.ones(
            size=[obs.shape[0], 1], dtype=torch.float32, device=obs.device
        )
        expand_obs = torch.cat((obs, virtual_t), 1)
        multiplier = self.pi(expand_obs)
        return multiplier
class FiniteHorizonFullPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy for finite-horizon.
    Input: observation, time step.
    Output: action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        self.act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.pre_horizon = kwargs["pre_horizon"]
        pi_sizes = [obs_dim] + list(hidden_sizes) + [self.act_dim * self.pre_horizon]

        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]).float())
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]).float())
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        return self.forward_all_policy(obs)[:, 0, :]

    def forward_all_policy(self, obs):
        actions = self.pi(obs).reshape(obs.shape[0], self.pre_horizon, self.act_dim)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(actions) \
                 + (self.act_high_lim + self.act_low_lim) / 2
        return action


# Stochastic Policy
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
        self.std_type = kwargs["std_type"]
        adapter_layers = kwargs.get("adapter_layers", None)
        adapter_start_layer = kwargs.get("adapter_start_layer", 0)
        add_bias = kwargs.get("adapter_add_bias", True)
        self._enable_adapter = kwargs.get("enable_adapter", False)
        self.adaption_ratio = kwargs.get("adaption_ratio", 0.1)
        self.consid_last_act = kwargs.get("consid_last_act", False)
        self._last_act = torch.zeros(1, act_dim, dtype=torch.float32)


        # mean and log_std are calculated by different MLP
        if self.std_type == "mlp_separated":
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
            self.mean = AMLP(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
            adapter_layers=adapter_layers,
            adapter_start_layer=adapter_start_layer,
            add_bias=add_bias,
            consider_last_act=self.consid_last_act,
            act_dim = act_dim 
        )
                
            self.log_std = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
        # mean and log_std are calculated by same MLP
        elif self.std_type == "mlp_shared":
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim * 2]
            self.policy =AMLP(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
            adapter_layers=adapter_layers,
            adapter_start_layer=adapter_start_layer,
            add_bias=add_bias,
            consider_last_act=self.consid_last_act,
            act_dim = act_dim 
        )
        # mean is calculated by MLP, and log_std is learnable parameter
        elif self.std_type == "parameter":
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
            self.mean = AMLP(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
            adapter_layers=adapter_layers,
            adapter_start_layer=adapter_start_layer,
            add_bias=add_bias,
            consider_last_act=self.consid_last_act,
            act_dim = act_dim 
        )
            self.log_std = nn.Parameter(-0.5*torch.ones(1, act_dim))

        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        if self._enable_adapter:
            if self.std_type == "mlp_separated":
                action_mean, adapter = self.mean(obs, self._last_act)
                action_std = torch.clamp(
                    self.log_std(obs), self.min_log_std, self.max_log_std
                ).exp()
                action_mean = action_mean + adapter.clamp(-1, 1)*self.adaption_ratio
            elif self.std_type == "mlp_shared":
                logits, logits_adapter = self.policy(obs, self._last_act)
                action_mean, action_log_std = torch.chunk(
                    logits, chunks=2, dim=-1
                )  # output the mean
                adapter,_ = torch.chunk(logits_adapter, chunks=2, dim=-1) # output the mean
                action_std = torch.clamp(
                    action_log_std, self.min_log_std, self.max_log_std
                ).exp()
                action_mean = action_mean + adapter.clamp(-1, 1)*self.adaption_ratio
            elif self.std_type == "parameter":
                action_mean, adapter = self.mean(obs, self._last_act)
                action_log_std = self.log_std + torch.zeros_like(action_mean)
                action_std = torch.clamp(
                    action_log_std, self.min_log_std, self.max_log_std
                ).exp()
                action_mean = action_mean + adapter.clamp(-1, 1)*self.adaption_ratio
        else:    
            if self.std_type == "mlp_separated":
                action_mean = self.mean(obs, self._last_act)
                action_std = torch.clamp(
                    self.log_std(obs), self.min_log_std, self.max_log_std
                ).exp()
            elif self.std_type == "mlp_shared":
                logits = self.policy(obs, self._last_act)
                action_mean, action_log_std = torch.chunk(
                    logits, chunks=2, dim=-1
                )  # output the mean
                action_std = torch.clamp(
                    action_log_std, self.min_log_std, self.max_log_std
                ).exp()
            elif self.std_type == "parameter":
                action_mean = self.mean(obs, self._last_act)
                action_log_std = self.log_std + torch.zeros_like(action_mean)
                action_std = torch.clamp(
                    action_log_std, self.min_log_std, self.max_log_std
                ).exp()
        self._last_act = action_mean
        return torch.cat((action_mean, action_std), dim=-1)

    def enable_adapter(self):
        self._enable_adapter = True
        if self.std_type == "mlp_separated":
            self.mean.enable_adapter()
        elif self.std_type == "mlp_shared":
            self.policy.enable_adapter()
        elif self.std_type == "parameter":
            self.mean.enable_adapter()
    def set_last_act(self, act):
        self._last_act = act
    def get_last_act(self):
        return self._last_act
    def reset_last_act(self,index):

        self._last_act[index, :] = 0.0
    
class ActionValue(nn.Module, Action_Distribution):
    """
    Approximated function of action-value function.
    Input: observation, action.
    Output: action-value.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [1],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class ActionValueDis(nn.Module, Action_Distribution):
    """
    Approximated function of action-value function for discrete action space.
    Input: observation.
    Output: action-value for all action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_num = kwargs["act_num"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.q = mlp(
            [obs_dim] + list(hidden_sizes) + [act_num],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        return self.q(obs)


class ActionValueDistri(nn.Module):
    """
    Approximated function of distributed action-value function.
    Input: observation.
    Output: parameters of action-value distribution.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [2],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        if "min_log_std"  in kwargs or "max_log_std" in kwargs:
            warnings.warn("min_log_std and max_log_std are deprecated in ActionValueDistri.")

    def forward(self, obs, act):
        logits = self.q(torch.cat([obs, act], dim=-1))
        value_mean, value_std = torch.chunk(logits, chunks=2, dim=-1)
        value_log_std = torch.nn.functional.softplus(value_std) 
        
        return torch.cat((value_mean, value_log_std), dim=-1)


class StochaPolicyDis(ActionValueDis, Action_Distribution):
    """
    Approximated function of stochastic policy for discrete action space.
    Input: observation.
    Output: parameters of action distribution.
    """

    pass


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
        self.v = mlp(
            [obs_dim] + list(hidden_sizes) + [1],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        v = self.v(obs)
        return torch.squeeze(v, -1)
