#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Soft Actor-Critic (SAC) algorithm
#  Reference: Haarnoja T, Zhou A, Abbeel P et al (2018) 
#             Soft actor-critic: off-policy maximum entropy deep reinforcement learning with a stochastic actor. 
#             ICML, Stockholm, Sweden.
#  Update: 2021-03-05, Yujie Yang: create SAC algorithm

__all__ = ["ApproxContainer", "SACPI"]

import time
from copy import deepcopy
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam

from gops.algorithm.base import AlgorithmBase, ApprBase
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.tensorboard_setup import tb_tags
from gops.utils.gops_typing import DataDict
from gops.utils.common_utils import get_apprfunc_dict, FreezeParameters


class ApproxContainer(ApprBase):
    """Approximate function container for SAC.

    Contains one policy and two action values.
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)
        # create q networks
        q_args = get_apprfunc_dict("value", **kwargs)
        self.q1: nn.Module = create_apprfunc(**q_args)
        self.q2: nn.Module = create_apprfunc(**q_args)
        self.pi_net = self.q1.pi_net
        self.q2.pi_net = self.pi_net

        
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        if kwargs["target_PI"]:
            self.pi_net_target = deepcopy(self.pi_net)   # use target pi
            self.q1_target.pi_net = self.pi_net_target
            self.q2_target.pi_net = self.pi_net_target
        else:
            self.q1_target.pi_net = self.pi_net   # use online pi
            self.q2_target.pi_net = self.pi_net

        # create policy network
        policy_args = get_apprfunc_dict("policy", **kwargs)
        self.policy: nn.Module = create_apprfunc(**policy_args)
        self.policy.pi_net = self.pi_net
        # self.policy_target = deepcopy(self.policy)
        # self.policy_target.pi_net = self.pi_net  # NOTE: use online pi? is it correct?

        # set target network gradients
        for p in self.q1_target.ego_paras():
            p.requires_grad = False
        for p in self.q2_target.ego_paras():
            p.requires_grad = False
        if kwargs["target_PI"]:
            for p in self.pi_net_target.parameters():
                p.requires_grad = False

        # create entropy coefficient
        self.log_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))

        # create optimizers
        self.q1_optimizer = Adam(self.q1.ego_paras(), lr=kwargs["value_learning_rate"])
        self.q2_optimizer = Adam(self.q2.ego_paras(), lr=kwargs["value_learning_rate"])
        self.policy_optimizer = Adam(
            self.policy.ego_paras(), lr=kwargs["policy_learning_rate"]
        )
        self.pi_optimizer = Adam(self.pi_net.parameters(), lr=kwargs["pi_learning_rate"])
        self.alpha_optimizer = Adam([self.log_alpha], lr=kwargs["alpha_learning_rate"])

        self.optimizer_dict = {
            "policy": self.policy_optimizer,
            "q1": self.q1_optimizer,
            "q2": self.q2_optimizer,
            "pi": self.pi_optimizer,
            "alpha": self.alpha_optimizer,
        }
        self.init_scheduler(**kwargs)


    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


class SACPI(AlgorithmBase):
    """Soft Actor-Critic (SAC) algorithm

    Paper: https://arxiv.org/abs/1801.01290

    :param float gamma: discount factor.
    :param float tau: param for soft update of target network.
    :param bool auto_alpha: whether to adjust temperature automatically.
    :param float alpha: initial temperature.
    :param Optional[float] target_entropy: target entropy for automatic
        temperature adjustment.
    """

    def __init__(
        self,
        index: int = 0,
        gamma: float = 0.99,
        tau: float = 0.005,
        auto_alpha: bool = True,
        alpha: float = 0.2,
        target_entropy: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        self.alpha = alpha
        if target_entropy is None:
            target_entropy = -kwargs["action_dim"]
        self.target_entropy = target_entropy
        self.target_PI = kwargs["target_PI"]
        self.per_flag = kwargs["buffer_name"].startswith("prioritized") # FIXME: hard code
        self.pred_reward = kwargs.get("pred_reward", False)
    @property
    def adjustable_parameters(self):
        return ("gamma", "tau", "auto_alpha", "alpha", "target_entropy")

    def _local_update(self, data: DataDict, iteration: int) -> dict:
        tb_info = self.__compute_gradient(data, iteration)
        self.__update(iteration)
        return tb_info

    def get_remote_update_info(
        self, data: DataDict, iteration: int
    ) -> Tuple[dict, dict]:
        tb_info = self.__compute_gradient(data, iteration)

        update_info = {
            "q1_grad": [p._grad for p in self.networks.q1.ego_paras()],
            "q2_grad": [p._grad for p in self.networks.q2.ego_paras()],
            "policy_grad": [p._grad for p in self.networks.policy.ego_paras()],
             "pi_grad": [p._grad for p in self.networks.pi_net.parameters()],
            "iteration": iteration,
        }
        if self.auto_alpha:
            update_info.update({"log_alpha_grad":self.networks.log_alpha.grad})

        return tb_info, update_info

    def _remote_update(self, update_info: dict):
        iteration = update_info["iteration"]
        q1_grad = update_info["q1_grad"]
        q2_grad = update_info["q2_grad"]
        policy_grad = update_info["policy_grad"]
        pi_grad = update_info["pi_grad"]

        for p, grad in zip(self.networks.q1.ego_paras(), q1_grad):
            p._grad = grad
        for p, grad in zip(self.networks.q2.ego_paras(), q2_grad):
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
        act_dist = self.networks.create_action_distributions(logits)
        new_act, new_logp = act_dist.rsample()
        data.update({"new_act": new_act, "new_logp": new_logp})

        self.networks.q1_optimizer.zero_grad()
        self.networks.q2_optimizer.zero_grad()
        self.networks.policy_optimizer.zero_grad()
        self.networks.pi_optimizer.zero_grad()
        loss_q, q1, q2, idx, td_err,rew_loss = self.__compute_loss_q(data)
        if loss_q.requires_grad:
            loss_q.backward() 


        with FreezeParameters([self.networks.q1.q, self.networks.q2.q]):

            loss_policy, entropy = self.__compute_loss_policy(data)
            loss_policy.backward()



        if self.auto_alpha:
            self.networks.alpha_optimizer.zero_grad()
            if self.networks.log_alpha.requires_grad:
                loss_alpha = self.__compute_loss_alpha(data)
                loss_alpha.backward()

        # calculate gradient norm
        q1_grad_norm = torch.norm( torch.cat([p.grad.flatten() for p in self.networks.q1.ego_paras() if p.grad is not None]))
        q2_grad_norm = torch.norm( torch.cat([p.grad.flatten() for p in self.networks.q2.ego_paras() if p.grad is not None]))
        policy_grad_norm = torch.norm( torch.cat([p.grad.flatten() for p in self.networks.policy.ego_paras() if p.grad is not None]))
        pi_grad_norm = torch.norm( torch.cat([p.grad.flatten() for p in self.networks.pi_net.parameters() if p.grad is not None]))
        tb_info = {
            tb_tags["loss_critic"]: loss_q.item(),
            tb_tags["loss_actor"]: loss_policy.item(),
            "SAC/critic_avg_q1-RL iter": q1.item(),
            "SAC/critic_avg_q2-RL iter": q2.item(),
            "SAC/entropy-RL iter": entropy.item(),
            "SAC/alpha-RL iter": self.__get_alpha(),
            "SAC/q_grad_norm": (q1_grad_norm+ q2_grad_norm).item()/2,
            "SAC/policy_grad_norm": policy_grad_norm.item(),
            "SAC/pi_grad_norm": pi_grad_norm.item(),
            tb_tags["alg_time"]: (time.time() - start_time) * 1000,
            "SAC/rew_loss": rew_loss.item() if rew_loss is not None else 0,
        }
        if self.per_flag:
            return tb_info, idx, td_err
        else:
            return tb_info

    def __compute_loss_q(self, data: DataDict):
        obs, act, rew, obs2, done = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        q1 = self.networks.q1(obs, act)
        q2 = self.networks.q2(obs, act)
        with torch.no_grad():
            next_logits = self.networks.policy(obs2)
            next_act_dist = self.networks.create_action_distributions(next_logits)
            next_act, next_logp = next_act_dist.rsample()
            next_q1 = self.networks.q1_target(obs2, next_act)
            next_q2 = self.networks.q2_target(obs2, next_act)
            next_q = torch.min(next_q1, next_q2)
            backup = rew + (1 - done) * self.gamma * (
                next_q - self.__get_alpha() * next_logp
            )
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()

        if self.per_flag:
            idx = data["idx"]
            td_err = (torch.abs(next_q1 - q1) + torch.abs(next_q2 - q2)) / 2
            # print("td_err_max", td_err.max().item())
            # print("td_err_min", td_err.min().item())
            per = td_err/2000 # TODO: 2000 is a hyperparameter
        else:
            idx = None
            per = None

        if data.get("reward_comps", None) is not None and self.pred_reward and obs.dim()<=2:  # need to learn reward component
            rew_comps = data["reward_comps"]
            rew_pred = self.networks.q1.predict_reward(obs, act)
            rew_loss = torch.mean((rew_pred - rew_comps) ** 2)
            loss_q1 += rew_loss
        else:
            rew_loss = None

        return loss_q1 + loss_q2, q1.detach().mean(), q2.detach().mean(), idx, per, rew_loss

    def __compute_loss_policy(self, data: DataDict):
        obs, new_act, new_logp = data["obs"], data["new_act"], data["new_logp"]
        obs = data.get("raw_obs", obs)
        q1 = self.networks.q1(obs, new_act)
        q2 = self.networks.q2(obs, new_act)
        loss_policy = (self.__get_alpha() * new_logp - torch.min(q1, q2)).mean()
        entropy = -new_logp.detach().mean()
        return loss_policy, entropy

    def __compute_loss_alpha(self, data: DataDict):
        new_logp = data["new_logp"]
        loss_alpha = (
            -self.networks.log_alpha * (new_logp.detach() + self.target_entropy).mean()
        )
        return loss_alpha

    def __update(self, iteration: int):
        self.networks.q1_optimizer.step()
        self.networks.q2_optimizer.step()
        self.networks.pi_optimizer.step()

        self.networks.policy_optimizer.step()

        if self.auto_alpha:
            self.networks.alpha_optimizer.step()

        with torch.no_grad():
            polyak = 1 - self.tau
            for p, p_targ in zip(
                self.networks.q1.ego_paras(), self.networks.q1_target.ego_paras()
            ):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
            for p, p_targ in zip(
                self.networks.q2.ego_paras(), self.networks.q2_target.ego_paras()
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
