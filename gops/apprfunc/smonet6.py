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
# Smonet 5 use mlp function to cal tau


__all__ = [
    "DetermPolicy",
    "FiniteHorizonPolicy",
    "StochaPolicy",
    "ActionValue",
    "ActionValueDis",
    "ActionValueDistri",
    "StateValue",
]

import numpy as np
import torch
import torch.nn as nn
from typing import Union, Tuple, List
from torch.nn import functional as F
from functorch import vmap, jacrev
from gops.utils.common_utils import get_activation_func, cal_ave_exec_time
from gops.utils.act_distribution_cls import Action_Distribution
from gops.apprfunc.mlp import mlp

class TauLayers(nn.Module):
    def __init__(self, in_features: int, out_features: int, seq_len, layer_num: int =2, hidden_dim: int = 64, layer_norm=False,scale = 10, bias=-3) -> None:
        super(TauLayers, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer_num = layer_num
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        sizes = [self.in_features] + [hidden_dim] * (layer_num - 1) + [out_features]
        self.activation = nn.Sigmoid()
        self.layers = mlp(sizes, nn.GELU, nn.Identity)
        self.layer_norm = layer_norm
        
        if self.layer_norm:
            self.ln = nn.LayerNorm(in_features)
        self.scale = scale
        self.bias = bias
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_norm:
            x = self.ln(x)
            
        x = self.cal_feature(x)
        x = self.layers(x)/self.scale + +self.bias
        x = self.activation(x)    
        return x
    
    def cal_feature(self, x: torch.Tensor) -> torch.Tensor:
        feature = x[:,-1:,:]
        feature = feature.reshape(-1, self.in_features)
        return feature

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, seq_len={}".format(
            self.in_features, self.out_features, self.seq_len
        )
class ConvFilter(nn.Module):
    def __init__(
        self,
        features: int,
        seq_len: int,
        kernel_size: int,
        loss_weight: float = 0.,
        tau_layer_num: int = 1, # TODO: realize multi-layer tau
        sample_interval: int = 0.1,
    ) -> None:
        super(ConvFilter, self).__init__()
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.loss_weight = loss_weight
        self.tau_layer = TauLayers(in_features=features, out_features=features, seq_len=seq_len, layer_num=tau_layer_num, scale=10, layer_norm=False,hidden_dim=64)
        self.loss_tau = 0
        if self.loss_weight > 0:
            self.register_full_backward_pre_hook(backward_hook)
        self.eval()
        self.need_regularization = True
        self.need_backward = False
        
    @staticmethod
    # @torch.jit.script
    def cal_output(tau: torch.Tensor, x: torch.Tensor, kernel_size: int, seq_len: int):
        den = torch.zeros_like(tau)
        for i in range(kernel_size):
            den = den + tau ** i
        nomalize_factor = 1/ den.detach()
        fliter_weight  = [nomalize_factor*tau ** (kernel_size - 1 - i) for i in range(kernel_size)]
        fliter_weight = torch.stack(fliter_weight, dim=2)

        output = []
        for i in range(seq_len - kernel_size + 1):
            output.append(torch.sum(fliter_weight[:,i,:,:]*x[:,i:i+kernel_size,:], dim=-2, keepdim=True))
        output = torch.cat(output, dim=-2)
        return output
    
    # @cal_ave_exec_time(print_interval=500) 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau = self.tau_layer(x.detach()) # tau = [B,D]
        tau = tau.unsqueeze(1) # tau = [B,1,D]
        tau = tau.expand(-1, self.seq_len-self.kernel_size+1, -1) # tau = [B,L-K+1,D]
        output = self.cal_output(tau,x,self.kernel_size,self.seq_len)
        if tau.requires_grad and self.training and self.loss_weight > 0 and self.need_regularization:
            self.update_loss_tau(tau)
        return output
    
    def update_loss_tau(self, tau: torch.Tensor):
        self.loss_tau +=  self.loss_weight*((1-tau)**2).squeeze(1).sum(-1).mean()
        self.need_backward = True

    def get_loss_tau(self):
        return self.loss_tau
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
    def add_regularization(self):
        self.need_regularization = True
    def remove_regularization(self):
        self.need_regularization = False 

def backward_hook(module, gout):
    if module.need_backward:
        module.loss_tau.backward(retain_graph=True)
        module.loss_tau = 0
        module.need_backward = False
    return gout
    


class FLinear(nn.Linear):
    def __init__(
        self,   
        in_features: int,   
        out_features: int,  
        bias: bool = True,
        loss_weight: float = 0.,   
        seq_len: int = 1,
        kernel_size: int = 1,
        tau_layer_num: int = 1,
    ) -> None:
        
        super(FLinear, self).__init__(   
            in_features=in_features,   
            out_features=out_features,  
            bias=bias,
        )

        self.kernel_size = kernel_size
        self.seq_len = seq_len
        self.seq_input = True
        if kernel_size == 1:
            pass
        else:
            assert kernel_size <= seq_len, "kernel_size should be less than seq_len"
            self.conv = ConvFilter(features=in_features, seq_len=seq_len, kernel_size=kernel_size, loss_weight=loss_weight, tau_layer_num=tau_layer_num)
        
    def forward(self, input):
        dim_size = input.dim()
        if self.kernel_size > 1 and dim_size > 2 and self.seq_input:
            input = self.conv(input)
        output = self.cat_forward(input)  
        return output
    
    def cat_forward(self, input):
        if self.seq_len == 1 or input.dim() <= 2:
            output = super(FLinear, self).forward(input)
        else:
            input_cur = input[:, -1:, :]
            input_pre = input[:, :-1, :]
            if self.seq_input:
                with torch.no_grad():
                    out_pre = super(FLinear, self).forward(input_pre)
                out_cur = super(FLinear, self).forward(input_cur)
                output = torch.cat((out_pre, out_cur), dim=1)
            else:
                output = super(FLinear, self).forward(input_cur)
        return output
      
    # def init_weights(self):
    #     nn.init.orthogonal_(self.weight, gain=np.sqrt(2))
    #     if self.bias is not None:
    #         nn.init.zeros_(self.bias)
    def freeze(self):
        # do not freeze the conv layer
        for param in self.parameters():
            param.requires_grad = False
            
        if hasattr(self, "conv"):
            self.conv.unfreeze()

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        
    def add_regularization(self):
        if hasattr(self, "conv"):
            self.conv.add_regularization()
    def remove_regularization(self):
        if hasattr(self, "conv"):
            self.conv.remove_regularization()
        
        
class FMLP(nn.Module):
    def __init__(self, sizes, kernel_size, activation,init_seq_len, output_activation=nn.Identity, loss_weight=0.,tau_layer_num=1):
        super(FMLP, self).__init__()
        layers = []
        seq_len = init_seq_len
        for j in range(len(sizes) - 1):
            act = activation if j < len(sizes) - 2 else output_activation
            layers += [ FLinear(in_features=sizes[j], out_features=sizes[j + 1],
                               kernel_size=kernel_size[j], loss_weight=loss_weight,seq_len=seq_len,tau_layer_num=tau_layer_num), act(),]
            seq_len = (seq_len - kernel_size[j] + 1)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    def freeze(self):
        for module in self.model.children():
            if isinstance(module, FLinear):
                module.freeze()
    def no_seq_len(self):
        for module in self.model.children():
            if isinstance(module, FLinear):
                module.seq_input = False
    def add_regularization(self):
        for module in self.model.children():
            if isinstance(module, FLinear):
                module.add_regularization()
    def remove_regularization(self):
        for module in self.model.children():
            if isinstance(module, FLinear):
                module.remove_regularization()



# def fmlp(sizes, kernel_size, activation,init_seq_len, output_activation=nn.Identity, loss_weight=0.,tau_layer_num=1):
#     layers = []
#     seq_len = init_seq_len
#     for j in range(len(sizes) - 1):
#         act = activation if j < len(sizes) - 2 else output_activation

#         layers += [ FLinear(in_features=sizes[j], out_features=sizes[j + 1],
#                            kernel_size=kernel_size[j], loss_weight=loss_weight,seq_len=seq_len,tau_layer_num=tau_layer_num), act(),]
        
#         seq_len = (seq_len - kernel_size[j] + 1)

#   return nn.Sequential(*layers)

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
    """
    Approximated function of deterministic policy.
    Input: observation.
    Output: action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"][1]  #FIXME: need to be updated
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        kernel_size = kwargs.get("kernel_size", [1] * (len(hidden_sizes) + 1))
        loss_weight = kwargs.get("loss_weight", 0.)
        init_seq_len = kwargs["obs_dim"][0]
        tau_layer_num = kwargs.get("tau_layer_num", 2)



        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = FMLP(
            pi_sizes,
            kernel_size,
            get_activation_func(kwargs["hidden_activation"]),
            init_seq_len= init_seq_len,
            output_activation = get_activation_func(kwargs["output_activation"]),
            loss_weight= loss_weight,
            tau_layer_num=tau_layer_num,
        )
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        # obs = obs.transpose(-1, -2)  # trans to (batch_size, obs_dim, seq_len)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
            self.pi(obs).squeeze(-2)
        ) + (self.act_high_lim + self.act_low_lim) / 2
        return action

class FiniteHorizonPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy for finite-horizon.
    Input: observation, time step.
    Output: action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"][1] + 1  # FIXME: need to be updated
        act_dim = kwargs["act_dim"]
        self.origin_obs_dim = obs_dim - act_dim -1
        hidden_sizes = kwargs["hidden_sizes"]
        kernel_size = kwargs.get("kernel_size", [1] * (len(hidden_sizes) + 1))
        loss_weight = kwargs.get("loss_weight", 0.)
        init_seq_len = kwargs["obs_dim"][0]
        tau_layer_num = kwargs.get("tau_layer_num", 2)


        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = FMLP(
            pi_sizes,
            kernel_size,
            get_activation_func(kwargs["hidden_activation"]),
            init_seq_len= init_seq_len,
            output_activation = get_activation_func(kwargs["output_activation"]),
            loss_weight= loss_weight,
            tau_layer_num=tau_layer_num,
        )
        act_high_lim = torch.from_numpy(kwargs["act_high_lim"]).float()
        act_low_lim = torch.from_numpy(kwargs["act_low_lim"]).float()
        self.register_buffer("act_high_lim", act_high_lim)
        self.register_buffer("act_low_lim", act_low_lim)
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs, virtual_t=1):
       # obs = obs.transpose(-1, -2)  # trans to (batch_size, obs_dim, seq_len)
        virtual_t = virtual_t * torch.ones(
            size=[obs.shape[0], obs.shape[1], 1], dtype=torch.float32, device=obs.device
        )
        expand_obs = torch.cat((obs, virtual_t), -1)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
            self.pi(expand_obs).squeeze(-2)
        ) + (self.act_high_lim + self.act_low_lim) / 2
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
        self.origin_obs_dim = obs_dim - act_dim
        self.act_dim = act_dim
        hidden_sizes = kwargs["hidden_sizes"]
        kernel_size = kwargs.get("kernel_size", [1] * (len(hidden_sizes) + 1))
        loss_weight = kwargs.get("loss_weight", 0.)
        self.std_type = kwargs["std_type"]
        init_seq_len = kwargs["seq_len"]
        tau_layer_num = kwargs.get("tau_layer_num", 2)
        self._is_freeze = False
        # mean and log_std are calculated by different MLP
        if self.std_type == "mlp_separated":
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
            self.mean = FMLP(
            pi_sizes,
            kernel_size,
            get_activation_func(kwargs["hidden_activation"]),
            init_seq_len,
            get_activation_func(kwargs["output_activation"]),
            loss_weight = loss_weight,
            tau_layer_num=tau_layer_num,

        )
            self.log_std = FMLP(
            pi_sizes,
            kernel_size,
            get_activation_func(kwargs["hidden_activation"]),
            init_seq_len,
            get_activation_func(kwargs["output_activation"]),
            loss_weight = loss_weight,
            tau_layer_num=tau_layer_num,
        )
        # mean and log_std are calculated by same MLP
        elif self.std_type == "mlp_shared":
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim * 2]
            self.policy = FMLP(
            pi_sizes,
            kernel_size,
            get_activation_func(kwargs["hidden_activation"]),
            init_seq_len,
            get_activation_func(kwargs["output_activation"]),
            loss_weight = loss_weight,
            tau_layer_num=tau_layer_num,
        )
        # mean is calculated by MLP, and log_std is learnable parameter
        elif self.std_type == "parameter":
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
            self.mean = FMLP(
            pi_sizes,
            kernel_size,
            get_activation_func(kwargs["hidden_activation"]),
            init_seq_len,
            get_activation_func(kwargs["output_activation"]),
            loss_weight = loss_weight,
            tau_layer_num=tau_layer_num,
        )
            self.log_std = nn.Parameter(torch.zeros(1, act_dim))

        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        act_high_lim = torch.from_numpy(kwargs["act_high_lim"]).float()
        act_low_lim = torch.from_numpy(kwargs["act_low_lim"]).float()
        self.register_buffer("act_high_lim", act_high_lim)
        self.register_buffer("act_low_lim", act_low_lim)
        self.action_distribution_cls = kwargs["action_distribution_cls"]
        self.need_regularization = False

    def forward(self, obs):
        # obs = obs.transpose(-1, -2)  # trans to (batch_size, obs_dim, seq_len)
        if self.std_type == "mlp_separated":
            action_mean = self.mean(obs).view(-1,self.act_dim)
            action_std = torch.clamp(
                self.log_std(obs).view(-1,self.act_dim), self.min_log_std, self.max_log_std
            ).exp()
        elif self.std_type == "mlp_shared":
            logits = self.policy(obs).view(-1, 2*self.act_dim)
            action_mean, action_log_std = torch.chunk(
                logits, chunks=2, dim=-1
            )  # output the mean
            action_std = torch.clamp(
                action_log_std, self.min_log_std, self.max_log_std
            ).exp()
        elif self.std_type == "parameter":
            action_mean = self.mean(obs).view(-1,self.act_dim)
            action_log_std = self.log_std + torch.zeros_like(action_mean)
            action_std = torch.clamp(
                action_log_std, self.min_log_std, self.max_log_std
            ).exp()

        return torch.cat((action_mean, action_std), dim=-1)
    
    def freeze(self):
        for module in self.modules():
            if isinstance(module, FMLP):
                module.freeze()
        self._is_freeze = True




    
    def add_regularization(self):
        self.need_regularization = True
        for module in self.modules():
            if isinstance(module, FMLP):
                module.add_regularization()

        

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
        kernel_size = kwargs.get("kernel_size", [1] * (len(hidden_sizes) + 1))
        loss_weight = kwargs.get("loss_weight", 0.)
        pi_sizes = [obs_dim + act_dim] + list(hidden_sizes) + [1]
        init_seq_len = kwargs["seq_len"]
        tau_layer_num = kwargs.get("tau_layer_num", 2)

        self.q = FMLP(
            pi_sizes,
            kernel_size,
            get_activation_func(kwargs["hidden_activation"]),
            init_seq_len,
            get_activation_func(kwargs["output_activation"]),
            loss_weight = loss_weight,
            tau_layer_num=tau_layer_num,
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs, act):
        # obs = obs.transpose(-1, -2)  # trans to (batch_size, obs_dim, seq_len)
        # expand_act = act.unsqueeze(-1).expand(-1, -1, obs.shape[-1])

        expand_act = act.unsqueeze(-2).expand(-1, obs.shape[-2],-1) # (batch_size, seq_len, obs_dim)
        input = torch.cat([obs, expand_act], dim=-1)

        q = self.q(input).view(-1, 1)
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
        kernel_size = kwargs.get("kernel_size", [1] * (len(hidden_sizes) + 1))
        loss_weight = kwargs.get("loss_weight", 0.)
        pi_sizes = [obs_dim + act_dim] + list(hidden_sizes) + [2]
        init_seq_len = kwargs["seq_len"]
        tau_layer_num = kwargs.get("tau_layer_num", 2)
        self.q = FMLP(
            pi_sizes,
            kernel_size,
            get_activation_func(kwargs["hidden_activation"]),
            init_seq_len,
            get_activation_func(kwargs["output_activation"]),
            loss_weight = loss_weight,
            tau_layer_num=tau_layer_num,
        )
        self.q.no_seq_len()
        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.denominator = max(abs(self.min_log_std), self.max_log_std)

    def forward(self, obs, act):
        if obs.dim() == 3:
            expand_act = act.unsqueeze(-2).expand(-1, obs.shape[-2],-1) # (batch_size, seq_len, obs_dim)
            input = torch.cat([obs, expand_act], dim=-1)
            logits = self.q(input).view(-1, 2)
        else:
            logits = self.q(torch.cat([obs, act], dim=-1))
        value_mean, value_std = torch.chunk(logits, chunks=2, dim=-1)
        value_log_std = torch.nn.functional.softplus(value_std) 
        return torch.cat((value_mean, value_log_std), dim=-1)
    
    def freeze(self):
        for module in self.modules():
            if isinstance(module, FMLP):
                module.freeze()


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
        obs_dim = kwargs["obs_dim"][1]  #FIXME: need to be updated
        hidden_sizes = kwargs["hidden_sizes"]
        kernel_size = kwargs.get("kernel_size", [1] * (len(hidden_sizes) + 1))
        loss_weight = kwargs.get("loss_weight", 0.)
        init_seq_len = kwargs["obs_dim"][0]
        tau_layer_num = kwargs.get("tau_layer_num", 2)

        pi_sizes = [obs_dim] + list(hidden_sizes) + [1]
        self.v = FMLP(
            pi_sizes,
            kernel_size,
            get_activation_func(kwargs["hidden_activation"]),
            init_seq_len,
            get_activation_func(kwargs["output_activation"]),
            loss_weight = loss_weight,
            tau_layer_num=tau_layer_num,
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        # obs = obs.transpose(-1, -2)
        v = self.v(obs)
        return torch.squeeze(v, -2).squeeze(-1)

    def get_tau_loss(self,input):
        self.forward(input)
        loss_tau = 0
        for layer in self.modules():
            if isinstance(layer, FLinear):
                loss_tau += layer.get_loss_tau()
        return loss_tau


if __name__ == '__main__':
    import torch
    from torch import nn
    from torch.autograd import Variable


    def test_FLinear_tau():
        # 定义输入数据
        input_data = torch.cat((torch.zeros(1, 3, 1), torch.zeros(1, 3, 1), torch.ones(1, 3, 1)), dim=2)

        # 定义 FLinear 类实例
        flinear = FLinear(in_channels=3, out_channels=1, kernel_size=3, tau=0.2)
        flinear.weight.data = torch.ones(2, 3, 3)
        flinear.bias.data = torch.zeros(2)
        # 计算输出
        output = flinear(input_data)
        # 检查输出形状是否正确
        assert output.shape == (1, 2, 1)
        # 检查 tau 参数是否正确设置
        assert flinear.tau.item() == torch.logit(torch.tensor(0.2))


    test_FLinear_tau()
