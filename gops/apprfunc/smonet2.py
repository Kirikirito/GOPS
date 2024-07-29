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
        self.weights_bias =10
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
        output = output[:,-self.seq_len:,:] # B, L, D
        output = output.unfold(1, self.roll_out_len, 1) # B, L-K+1, D, K
        output[...,-1] = output[...,-1] + self.weights_bias 
        output = F.softmax(output, dim=-1) 
        return output

class ConvFilter(nn.Module):
    def __init__(
        self,
        features: int,
        seq_len: int,
        kernel_size: int,
        loss_weight: float,
        tau_layer_num: int,
        sample_interval: int = 0.1,
    ) -> None:
        super(ConvFilter, self).__init__()
        self.seq_len = seq_len
        self.loss_weight = loss_weight
        self.features = features
        self.weights_layer = TransformerEncoder(
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
        
    # @torch.jit.script_method
    # @ cal_ave_exec_time(print_interval=500)
    def cal_output(self, attn_weights: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        '''
        Calculate the output of the convolutional filter.
        Args:
            attn_weights: The weights of the convolutional filter, shape: B, L-K+1, D,K
            x: The input data, shape: B, L, D
        '''

        x = x.unfold(1, self.kernel_size, 1)  # B, L-K+1, D, K
        output = torch.einsum("bldk,bldk->bld", attn_weights, x)  # B, L-K+1, D
        # # seq_range: [L-1,L-2,...,0]
        # seq_range = torch.arange(self.kernel_size-1, -1, -1, device=x.device)
        # powers = tau.unsqueeze(-1).pow(seq_range) # B,D,L
        # normalize_factor =  powers.sum(dim=2) # B,D
        # filter_weight = powers / normalize_factor.unsqueeze(-1) # B,D,L
        # x = x.transpose(1, 2).contiguous()  # B,D,L
        # filter_weight = filter_weight.view(-1, 1, self.kernel_size)  # BD, 1, 
        # # tau: B,D, filter_weight: B,D,L, x: B,L,D
        # output = F.conv1d(x.view(1, -1, self.seq_len), filter_weight, stride=1, groups=filter_weight.shape[0])
        # output = output.view(-1, self.features, self.seq_len - self.kernel_size + 1).transpose(1, 2)  # B, L, D
        return output
    
    # @cal_ave_exec_time(print_interval=500) 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # only the last sequential is used for backward
        # x = torch.cat([x[:,:self.seq_len-1,:].detach(),x[:,-1:,:]], dim=1) # TODO: necessary to detach?
        with torch.no_grad():
            feature = self.cal_feature(x)
        
        attn_weights = self.weights_layer(feature)
        # to create a new graph for tau's regularization
        attn_weights2 = self.weights_layer(feature)
        output = self.cal_output(attn_weights,x)
        if attn_weights2.requires_grad and self.training and self.loss_weight > 0:
            self.update_loss_tau(attn_weights2)
            # print(tau.max().item(), tau.min().item())
        return output
    
    #@cal_ave_exec_time(print_interval=50000)
    # @torch.jit.script_method
    # @cal_ave_exec_time(print_interval=500)
    def cal_feature(self, x: torch.Tensor) -> torch.Tensor:
        amp = torch.fft.rfft(x, dim=1).abs() # B,L,D, L for seq_len
        feature = torch.cat((x,amp), dim=-2) # B,2L,D
        return feature
        


    def update_loss_tau(self, attn_weights):
        # MSE loss for tau
        self.loss_tau += self.loss_weight * nn.functional.mse_loss(attn_weights*self.kernel_size, torch.ones_like(attn_weights), reduction="mean")
        

    def get_loss_tau(self):
        return self.loss_tau
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

def backward_hook(module, gout):
    if module.loss_tau != 0:
        module.loss_tau.backward(retain_graph=False)
        module.loss_tau = 0
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
        if self.seq_len == 1 or input.dim() == 2:
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
        self.std_type = kwargs["std_type"]
        init_seq_len = kwargs["seq_len"]
        tau_layer_num = kwargs.get("tau_layer_num", 1)

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
