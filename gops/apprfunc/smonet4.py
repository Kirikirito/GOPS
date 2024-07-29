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


def init_weights(m):
    for child in m.children():
        if isinstance(child, nn.Linear):
            nn.init.orthogonal_(child.weight, gain=np.sqrt(2))
            nn.init.zeros_(child.bias)
        elif isinstance(child, nn.BatchNorm1d):
            nn.init.constant_(child.weight, 1)
            nn.init.constant_(child.bias, 0)
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0., activation="relu", 
                 seq_len = 1,roll_out_len=1):
        super(TransformerEncoder, self).__init__()
        self.seq_len = seq_len
        self.roll_out_len = roll_out_len
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True),
            num_encoder_layers,
        )
        self.softplus = nn.Softplus()
        self.div_base = 10.
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
    def forward(self, src, mask=None):
        src = self.positional_encoding(src)
        output = self.transformer(src, mask)
        output, _ = self.multihead_attn(output, output, output)
        # get the weights of the last sequence
        output = self.recal_weights(output)


        return output
    def recal_weights(self, output):
        # add bias to the last sequence for the weights
        output = output[:,-1:,:] # B, L, D
        output = self.softplus(output)/self.div_base
        output = torch.clamp(output, min=1e-2, max=1)
        return output

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
        self.loss_weight = loss_weight
        self.features = features
        self.tau_layer = TransformerEncoder(
            d_model=features,
            nhead=8,
            num_encoder_layers=tau_layer_num,
            dim_feedforward=features * 4,
            dropout=0.0,
            activation="relu",
            seq_len=seq_len,
            roll_out_len=kernel_size,
        )
        self.kernel_size = kernel_size
        self.loss_tau = 0
        if self.loss_weight > 0:
            self.register_full_backward_pre_hook(backward_hook)
        self.eval()
        self.tau_coff = 1
        self.sample_interval = sample_interval
        self.tau_upper_bound = 1.0
        self.tau_increment = 0
        self.tau_loss_fn = nn.MSELoss(reduction="mean")
        self.need_regularization = True
        self.need_backward = False
        
    # @torch.jit.script_method
    # @ cal_ave_exec_time(print_interval=500)
    def cal_output(self, tau: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        '''
        Calculate the output of the convolutional filter.
        Args:
            tau: The filter weights, shape: B, 1, D
            x: The input data, shape: B, L, D
        '''

        x = x.unfold(1, self.kernel_size, 1)  # B, L-K+1, D, K
        seq_range = torch.arange(self.kernel_size-1, -1, -1, device=x.device) # K
        powers = tau.transpose(1, 2).pow(seq_range) # B, D, K
        normalize_factor = powers.sum(dim=2) # B, D
        filter_weight = powers / normalize_factor.unsqueeze(-1) # B, D, K
        output = torch.einsum("bdk,bldk->bld", filter_weight, x)  # B, L-K+1, D

        return output
    
    # @cal_ave_exec_time(print_interval=500) 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # only the last sequential is used for backward
        # x = torch.cat([x[:,:self.seq_len-1,:].detach(),x[:,-1:,:]], dim=1) # TODO: necessary to detach?
        with torch.no_grad():
            feature = self.cal_feature(x)
        
        tau = self.tau_layer(feature)
        # to create a new graph for tau's regularization
        # tau2 = self.tau_layer(feature)
        output = self.cal_output(tau,x)
        if tau.requires_grad and self.training and self.loss_weight > 0 and self.need_regularization:
            self.update_loss_tau(tau)
        return output
    
    #@cal_ave_exec_time(print_interval=50000)
    # @torch.jit.script_method
    # @cal_ave_exec_time(print_interval=500)
    def cal_feature(self, x: torch.Tensor) -> torch.Tensor:
        amp = torch.fft.rfft(x, dim=1).abs() # B,L,D, L for seq_len
        feature = torch.cat((x[:,-1:,:],amp), dim=-2) # B,L+1,D
        return feature
        


    def update_loss_tau(self, tau: torch.Tensor):
        # MSE loss for tau
        self.loss_tau += self.loss_weight * self.tau_loss_fn(tau, torch.ones_like(tau))*self.features
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
        obs_dim = kwargs["obs_dim"][1]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        kernel_size = kwargs.get("kernel_size", [1] * (len(hidden_sizes) + 1))
        loss_weight = kwargs.get("loss_weight", 0.)
        init_seq_len = kwargs["obs_dim"][0]
        tau_layer_num = kwargs.get("tau_layer_num", 1)



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
        obs_dim = kwargs["obs_dim"][1] + 1
        act_dim = kwargs["act_dim"]
        self.origin_obs_dim = obs_dim - act_dim -1
        hidden_sizes = kwargs["hidden_sizes"]
        kernel_size = kwargs.get("kernel_size", [1] * (len(hidden_sizes) + 1))
        loss_weight = kwargs.get("loss_weight", 0.)
        init_seq_len = kwargs["obs_dim"][0]
        tau_layer_num = kwargs.get("tau_layer_num", 1)


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
        self.loss_weight = loss_weight*0.005
        self.std_type = kwargs["std_type"]
        init_seq_len = kwargs["seq_len"]
        tau_layer_num = kwargs.get("tau_layer_num", 1)
        self.jacobian_norm = torch.tensor(0.0)
        self._is_freeze = False
        self._need_cal_jacobian = True
        self.register_full_backward_pre_hook(policy_prebak_hook)

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
        self.need_backward = False 
        self.appro_jacobian = False   

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
        if self.training and self.loss_weight > 0  and self._need_cal_jacobian and action_mean.requires_grad and self.need_regularization:
            self.cal_jacobian_norm(obs)

        return torch.cat((action_mean, action_std), dim=-1)
    
    def freeze(self):
        for module in self.modules():
            if isinstance(module, FMLP):
                module.freeze()
        self._is_freeze = True
    def cal_jacobian_norm(self, obs):
        self.need_backward = True
        obs.requires_grad = True
        self._need_cal_jacobian = False
        if obs.dim() == 3:
            obs = obs[:, -1, :] # B, D
        if self.appro_jacobian:
            out1 = self.forward(obs)
            delta = 1e-6 + torch.randn_like(obs) * 1e-6  # B, 2D
            delta_norm = torch.norm(delta, dim=1, p=2)
            out2 = self.forward(obs + delta)
            norm= (torch.norm((out2 - out1), dim=-1, p=1)/delta_norm)
            #print('norm_mean',norm.mean())
            #print('norm_max',norm.max())

            # jacobi = (out2 - out1) / delta
            # norm = torch.norm(jacobi[:,0,:self.act_dim], dim=-1, p=1).mean()
            norm = norm.mean()
        else:
            jacobi = vmap(jacrev(self.forward))(obs)
            norm = torch.norm(jacobi[:,0,:self.act_dim,:], dim=-1, p=1).mean()
        self.jacobian_norm = norm*self.loss_weight
        self._need_cal_jacobian = True



    
    def add_regularization(self):
        self.need_regularization = True
        for module in self.modules():
            if isinstance(module, FMLP):
                module.add_regularization()

def policy_prebak_hook(module, gout):
    if module.need_backward:
        module.jacobian_norm.backward(retain_graph=True)
        module.jacobian_norm = 0
        module.need_backward = False
    return gout
        

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
        tau_layer_num = kwargs.get("tau_layer_num", 1)

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
        tau_layer_num = kwargs.get("tau_layer_num", 1)
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
        obs_dim = kwargs["obs_dim"][1]
        hidden_sizes = kwargs["hidden_sizes"]
        kernel_size = kwargs.get("kernel_size", [1] * (len(hidden_sizes) + 1))
        loss_weight = kwargs.get("loss_weight", 0.)
        init_seq_len = kwargs["obs_dim"][0]
        tau_layer_num = kwargs.get("tau_layer_num", 1)

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
