#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Action Distributions
#  Update Date: 2021-03-10, Yujie Yang: Revise Codes


import torch
import scipy.integrate as integrate
import numpy as np

EPS = 1e-6


class TanhGaussDistribution:
    def __init__(self, logits):
        self.logits = logits
        self.mean, self.std = torch.chunk(logits, chunks=2, dim=-1)
        self.gauss_distribution = torch.distributions.Independent(
            base_distribution=torch.distributions.Normal(self.mean, self.std),
            reinterpreted_batch_ndims=1,
        )
        self.act_high_lim = torch.tensor([1.0])
        self.act_low_lim = torch.tensor([-1.0])

    def sample(self):
        action = self.gauss_distribution.sample()
        action_limited = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
            action
        ) + (self.act_high_lim + self.act_low_lim) / 2
        log_prob = (
            self.gauss_distribution.log_prob(action)
            - torch.log(1 + EPS - torch.pow(torch.tanh(action), 2)).sum(-1)
            - torch.log((self.act_high_lim - self.act_low_lim) / 2).sum(-1)
        )
        return action_limited, log_prob

    def rsample(self):
        action = self.gauss_distribution.rsample()
        action_limited = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
            action
        ) + (self.act_high_lim + self.act_low_lim) / 2
        log_prob = (
            self.gauss_distribution.log_prob(action)
            - torch.log(1 + EPS - torch.pow(torch.tanh(action), 2)).sum(-1)
            - torch.log((self.act_high_lim - self.act_low_lim) / 2).sum(-1)
        )
        return action_limited, log_prob

    def log_prob(self, action_limited) -> torch.Tensor:
        action = torch.atanh(
            (1 - EPS)
            * (2 * action_limited - (self.act_high_lim + self.act_low_lim))
            / (self.act_high_lim - self.act_low_lim)
        )
        log_prob = self.gauss_distribution.log_prob(action) - torch.log(
            (self.act_high_lim - self.act_low_lim)
            * (1 + EPS - torch.pow(torch.tanh(action), 2))
        ).sum(-1)
        return log_prob
        
    def entropy(self):
        gauss_entropy = self.gauss_distribution.entropy().detach().numpy()[0]
        entropy_tanh = 0
        mus = self.mean.detach().numpy()[0]
        stds= self.std.detach().numpy()[0]
        bound = ((self.act_high_lim - self.act_low_lim)/2).detach().numpy()

        def func(x):
            y = -2/(np.sqrt(2*np.pi))*np.exp(-(x)**2/(2))*np.log(np.cosh(sigma*x+mu))
            return y

        for i in range(len(mus)):
            a = bound[i]
            mu = mus[i]
            sigma = stds[i]
            entropy_tanh = entropy_tanh + integrate.quad(func, -6, 6)[0] + np.log(a)

        entropy_tanh = entropy_tanh + gauss_entropy
        # print(entropy_tanh)
        if np.isnan(entropy_tanh):
            entropy_tanh = - len(mus)

        return entropy_tanh


    def mode(self):
        return (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(self.mean) + (
            self.act_high_lim + self.act_low_lim
        ) / 2

    def kl_divergence(self, other: "GaussDistribution") -> torch.Tensor:
        return torch.distributions.kl.kl_divergence(
            self.gauss_distribution, other.gauss_distribution
        )


class GaussDistribution:
    def __init__(self, logits):
        self.logits = logits
        self.mean, self.std = torch.chunk(logits, chunks=2, dim=-1)
        self.gauss_distribution = torch.distributions.Independent(
            base_distribution=torch.distributions.Normal(self.mean, self.std),
            reinterpreted_batch_ndims=1,
        )
        self.act_high_lim = torch.tensor([1.0])
        self.act_low_lim = torch.tensor([-1.0])

    def sample(self):
        action = self.gauss_distribution.sample()
        log_prob = self.gauss_distribution.log_prob(action)
        return action, log_prob

    def rsample(self):
        action = self.gauss_distribution.rsample()
        log_prob = self.gauss_distribution.log_prob(action)
        return action, log_prob

    def log_prob(self, action) -> torch.Tensor:
        log_prob = self.gauss_distribution.log_prob(action)
        return log_prob

    def entropy(self):
        return self.gauss_distribution.entropy()

    def mode(self):
        return torch.clamp(self.mean, self.act_low_lim, self.act_high_lim)

    def kl_divergence(self, other: "GaussDistribution") -> torch.Tensor:
        return torch.distributions.kl.kl_divergence(
            self.gauss_distribution, other.gauss_distribution
        )


class CategoricalDistribution:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits
        self.cat = torch.distributions.Categorical(logits=logits)

    def sample(self):
        action = self.cat.sample()
        log_prob = self.log_prob(action)
        return action, log_prob

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        if action.dim() > 1:
            action = action.squeeze(1)
        return self.cat.log_prob(action)

    def entropy(self):
        return self.cat.entropy()

    def mode(self):
        return torch.argmax(self.logits, dim=-1)

    def kl_divergence(self, other: "CategoricalDistribution"):
        return torch.distributions.kl.kl_divergence(self.cat, other.cat)


class DiracDistribution:
    def __init__(self, logits):
        self.logits = logits

    def sample(self):
        return self.logits, torch.tensor([0.0])

    def mode(self):
        return self.logits


class ValueDiracDistribution:
    def __init__(self, logits):
        self.logits = logits

    def sample(self):
        return torch.argmax(self.logits, dim=-1), torch.tensor([0.0])

    def mode(self):
        return torch.argmax(self.logits, dim=-1)
