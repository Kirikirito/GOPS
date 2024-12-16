#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Distributed Soft Actor-Critic (DSAC) algorithm
#  Reference: Duan J, Guan Y, Li SE et al (2021) 
#             Distributional soft actor-critic: off-policy reinforcement learning 
#             for addressing value estimation errors. 
#             IEEE Transactions on Neural Network and Learning Systems 33(11): 6584-6598.
#  Update: 2021-03-05, Ziqing Gu: create DSAC algorithm
#  Update: 2021-03-05, Wenxuan Wang: debug DSAC algorithm

__all__ = ["ApproxContainer", "DSACPI"]

import time
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam

from gops.algorithm.base import AlgorithmBase, ApprBase
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.tensorboard_setup import tb_tags
from gops.utils.gops_typing import DataDict
from gops.utils.common_utils import get_apprfunc_dict, FreezeParameters


class ApproxContainer(ApprBase):
    """Approximate function container for DSAC.

    Contains one policy and one action value.
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)
        # create q networks
        q_args = get_apprfunc_dict("value", **kwargs)
        self.q: nn.Module = create_apprfunc(**q_args)
        self.pi_net = self.q.pi_net
        self.q_target = deepcopy(self.q)
        if kwargs["target_PI"]:
            self.pi_net_target = deepcopy(self.pi_net)   # use target pi
            self.q_target.pi_net = self.pi_net_target
        else:
            self.q_target.pi_net = self.pi_net   # use online pi
        # create policy network
        policy_args = get_apprfunc_dict("policy", **kwargs)
        self.policy: nn.Module = create_apprfunc(**policy_args)
        self.policy.pi_net = self.pi_net
        self.policy_target = deepcopy(self.policy)
        self.policy_target.pi_net = self.pi_net  # NOTE: use online pi? is it correct?

        # set target network gradients
        for p in self.policy_target.ego_paras():
            p.requires_grad = False
        for p in self.q_target.ego_paras():
            p.requires_grad = False
        if kwargs["target_PI"]:
            for p in self.pi_net_target.parameters():
                p.requires_grad = False

        # create entropy coefficient
        self.log_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))

        # create optimizers
        self.q_optimizer = Adam(self.q.ego_paras(), lr=kwargs["value_learning_rate"])
        self.policy_optimizer = Adam(
            self.policy.ego_paras(), lr=kwargs["policy_learning_rate"]
        )
        self.pi_optimizer = Adam(self.pi_net.parameters(), lr=kwargs["pi_learning_rate"])
        self.alpha_optimizer = Adam([self.log_alpha], lr=kwargs["alpha_learning_rate"])

        self.optimizer_dict = {
            "policy": self.policy_optimizer,
            "q": self.q_optimizer,
            "pi": self.pi_optimizer,
            "alpha": self.alpha_optimizer,
        }
        self.init_scheduler(**kwargs)


    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


class DSACPI(AlgorithmBase):
    """DSAC algorithm with PI net
    Paper: https://arxiv.org/pdf/2001.02811

    :param float gamma: discount factor.
    :param float tau: param for soft update of target network.
    :param bool auto_alpha: whether to adjust temperature automatically.
    :param float alpha: initial temperature.
    :param float TD_bound: the bound of temporal difference.
    :param bool bound: whether to bound the q value.
    :param float delay_update: delay update steps for actor.
    :param Optional[float] target_entropy: target entropy for automatic
        temperature adjustment.
    """

    def __init__(self, index=0, **kwargs):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.gamma = kwargs["gamma"]
        self.tau = kwargs["tau"]
        self.target_entropy = -kwargs["action_dim"]
        self.auto_alpha = kwargs["auto_alpha"]
        self.alpha = kwargs.get("alpha", 0.2)
        self.bound = kwargs["bound"]
        self.td_bound = kwargs.get("TD_bound", 10)
        self.delay_update = kwargs["delay_update"]
        self.target_PI = kwargs["target_PI"]
        self.per_flag = kwargs["buffer_name"].startswith("prioritized") # FIXME: hard code
        self.pred_reward = kwargs.get("pred_reward", False)
        

    @property
    def adjustable_parameters(self):
        return (
            "gamma",
            "tau",
            "auto_alpha",
            "alpha",
            "bound",
            "delay_update",
        )

    def _local_update(self, data: DataDict, iteration: int) -> dict:
        tb_info = self.__compute_gradient(data, iteration)
        self.__update(iteration)
        return tb_info

    def get_remote_update_info(
        self, data: DataDict, iteration: int
    ) -> Tuple[dict, dict]:
        tb_info = self.__compute_gradient(data, iteration)

        update_info = {
            "q_grad": [p._grad for p in self.networks.q.ego_paras()],
            "policy_grad": [p._grad for p in self.networks.policy.ego_paras()],
             "pi_grad": [p._grad for p in self.networks.pi_net.parameters()],
            "iteration": iteration,
        }
        if self.auto_alpha:
            update_info.update({"log_alpha_grad":self.networks.log_alpha.grad})

        return tb_info, update_info

    def _remote_update(self, update_info: dict):
        iteration = update_info["iteration"]
        q_grad = update_info["q_grad"]
        policy_grad = update_info["policy_grad"]
        pi_grad = update_info["pi_grad"]

        for p, grad in zip(self.networks.q.ego_paras(), q_grad):
            p._grad = grad

        for p, grad in zip(self.networks.policy.ego_paras(), policy_grad):
            p._grad = grad
        for p, grad in zip(self.networks.pi_net.parameters(), pi_grad):
            p._grad = grad
        if self.auto_alpha:
            self.networks.log_alpha._grad = update_info["log_alpha_grad"]

        self.__update(iteration)

    def __get_alpha(self, requires_grad: bool = False):
        if self.auto_alpha:
            alpha = self.networks.log_alpha.exp()
            if requires_grad:
                return alpha
            else:
                return alpha.item()
        else:
            return self.alpha

    def __compute_gradient(self, data: DataDict, iteration: int):
        start_time = time.time()

        obs = data["obs"]
        logits = self.networks.policy(obs)
        policy_mean = torch.tanh(logits[..., 0]).mean().item()
        policy_std = logits[..., 1].mean().item()

        act_dist = self.networks.create_action_distributions(logits)
        new_act, new_log_prob = act_dist.rsample()
        data.update({"new_act": new_act, "new_log_prob": new_log_prob})

        self.networks.q_optimizer.zero_grad()

        self.networks.policy_optimizer.zero_grad()
        self.networks.pi_optimizer.zero_grad()
        loss_q, q, std,idx, td_err = self.__compute_loss_q(data)
        if loss_q.requires_grad:
            loss_q.backward() 

        with FreezeParameters([self.networks.q.q,]):
            loss_policy, entropy = self.__compute_loss_policy(data)
            loss_policy.backward()



        if self.auto_alpha:
            self.networks.alpha_optimizer.zero_grad()
            if self.networks.log_alpha.requires_grad:
                loss_alpha = self.__compute_loss_alpha(data)
                loss_alpha.backward()

        # calculate gradient norm
        q1_grad_norm = torch.norm( torch.cat([p.grad.flatten() for p in self.networks.q.ego_paras() if p.grad is not None]))
        policy_grad_norm = torch.norm( torch.cat([p.grad.flatten() for p in self.networks.policy.ego_paras() if p.grad is not None]))
        pi_grad_norm = torch.norm( torch.cat([p.grad.flatten() for p in self.networks.pi_net.parameters() if p.grad is not None]))
        tb_info = {
            "DSAC/critic_avg_q-RL iter": q.item(),
            "DSAC/critic_avg_std-RL iter": std.item(),
            tb_tags["loss_actor"]: loss_policy.item(),
            "DSAC/policy_mean-RL iter": policy_mean,
            "DSAC/policy_std-RL iter": policy_std,
            "DSAC/entropy-RL iter": entropy.item(),
            "DSAC/alpha-RL iter": self.__get_alpha(),
            "DSAC2/q_grad_norm": q1_grad_norm.item(),
            "DSAC2/policy_grad_norm": policy_grad_norm.item(),
            "DSAC2/pi_grad_norm": pi_grad_norm.item(),
            tb_tags["alg_time"]: (time.time() - start_time) * 1000,
        }
        if self.per_flag:
            return tb_info, idx, td_err
        else:
            return tb_info

    def __q_evaluate(self, obs, act, qnet, use_min=False):
        StochaQ = qnet(obs, act)
        mean, std = StochaQ[..., 0], StochaQ[..., -1]
        normal = Normal(torch.zeros_like(mean), torch.ones_like(std))
        if use_min:
            z = -torch.abs(normal.sample())
        else:
            z = normal.sample()
            z = torch.clamp(z, -3, 3)
        q_value = mean + torch.mul(z, std)
        return mean, std, q_value

    def __compute_loss_q(self, data: DataDict):
        obs, act, rew, obs2, done = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        noise_obs = obs
        noise_obs2 = obs2
        obs = data.get("raw_obs", obs)
        obs2 = data.get("raw_obs2", obs2)
        if self.per_flag:
            weight = data["weight"]
        else:
            weight = 1.0
        with torch.no_grad():
            logits_2 = self.networks.policy_target(noise_obs2)
            act2_dist = self.networks.create_action_distributions(logits_2)
            act2, log_prob_act2 = act2_dist.rsample()
        q, q_std, q_sample = self.__q_evaluate(obs, act, self.networks.q, use_min=False)
        with torch.no_grad():
            _, _, q_next_sample = self.__q_evaluate(
                obs2, act2, self.networks.q_target, use_min=False
            )
        target_q, target_q_bound = self.__compute_target_q(
            rew,
            done,
            q.detach(),
            q_std.detach(),
            q_next_sample.detach(),
            log_prob_act2.detach(),
        )
        if self.bound:
            q_loss = torch.mean(
                torch.pow(q - target_q, 2) / (2 * torch.pow(q_std.detach(), 2))
                + torch.pow(q.detach() - target_q_bound, 2) / (2 * torch.pow(q_std, 2))
                + torch.log(q_std)
            )
        else:
            q_loss = -Normal(q, q_std).log_prob(target_q).mean()
        if self.per_flag:
            idx = data["idx"]
            td_err = torch.abs(target_q - q)
            # print("td_err_max", td_err.max().item())
            # print("td_err_min", td_err.min().item())
            per = td_err/2000 # TODO: 2000 is a hyperparameter
        else:
            idx = None
            per = None
        if data.get("reward_comps", None) is not None and self.pred_reward and obs.dim()<=2:  # need to learn reward component
            rew_comps = data["reward_comps"]
            rew_pred = self.networks.q.predict_reward(obs, act)
            rew_loss = torch.mean((rew_pred - rew_comps) ** 2)
            q_loss += rew_loss
        else:
            rew_loss = None
        return q_loss, q.detach().mean(), q_std.detach().mean(), idx, per

    def __compute_target_q(self, r, done, q, q_std, q_next, log_prob_a_next):
        target_q = r + (1 - done) * self.gamma * (
            q_next - self.__get_alpha() * log_prob_a_next
        )
        td_bound = self.td_bound 
        difference = torch.clamp(target_q - q, -td_bound, td_bound)
        target_q_bound = q + difference
        return target_q.detach(), target_q_bound.detach()

    def __compute_loss_policy(self, data: DataDict):
        obs, new_act, new_log_prob = data["obs"], data["new_act"], data["new_log_prob"]
        obs = data.get("raw_obs", obs)
        q, _, _ = self.__q_evaluate(obs, new_act, self.networks.q, use_min=False)
        loss_policy = (self.__get_alpha() * new_log_prob - q).mean()
        entropy = -new_log_prob.detach().mean()
        return loss_policy, entropy

    def __compute_loss_alpha(self, data: DataDict):
        new_log_prob = data["new_log_prob"]
        loss_alpha = (
            -self.networks.log_alpha
            * (new_log_prob.detach() + self.target_entropy).mean()
        )
        return loss_alpha

    def __update(self, iteration: int):
        self.networks.q_optimizer.step()
        self.networks.pi_optimizer.step()

        if iteration % self.delay_update == 0:
            self.networks.policy_optimizer.step()

            if self.auto_alpha:
                self.networks.alpha_optimizer.step()

            with torch.no_grad():
                polyak = 1 - self.tau
                for p, p_targ in zip(
                    self.networks.q.ego_paras(), self.networks.q_target.ego_paras()
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
                for p, p_targ in zip(
                    self.networks.policy.ego_paras(),
                    self.networks.policy_target.ego_paras(),
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

                if self.target_PI:
                    for p, p_targ in zip(
                        self.networks.pi_net.parameters(),
                        self.networks.pi_net_target.parameters(),
                    ):
                        p_targ.data.mul_(polyak)
                        p_targ.data.add_((1 - polyak) * p.data)