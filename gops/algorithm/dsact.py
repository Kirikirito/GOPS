#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Distributed Soft Actor-Critic (DSAC) algorithm
#  Reference: Duan J, Guan Y, Li S E, et al.
#             Distributional soft actor-critic: Off-policy reinforcement learning
#             for addressing value estimation errors[J].
#             IEEE transactions on neural networks and learning systems, 2021.
#  Update: 2021-03-05, Ziqing Gu: create DSAC algorithm
#  Update: 2021-03-05, Wenxuan Wang: debug DSAC algorithm

__all__=["ApproxContainer","DSACT"]
import time
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam
from torch.nn.functional import huber_loss
from copy import deepcopy
from gops.algorithm.base import AlgorithmBase, ApprBase
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.tensorboard_setup import tb_tags
from gops.utils.gops_typing import DataDict
from gops.utils.common_utils import get_apprfunc_dict, TimePerfRecorder, cal_ave_exec_time, FreezeParameters


class ApproxContainer(ApprBase):
    """Approximate function container for DSAC.

    Contains one policy and one action value.
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)
        # create q networks
        q_args = get_apprfunc_dict("value", **kwargs)
        self.q1: nn.Module = create_apprfunc(**q_args)
        self.q2: nn.Module = create_apprfunc(**q_args)
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)

        # create policy network
        policy_args = get_apprfunc_dict("policy", **kwargs)
        self.policy: nn.Module = create_apprfunc(**policy_args)
        self.policy_target = deepcopy(self.policy)

        # set target network gradients
        for p in self.policy_target.parameters():
            p.requires_grad = False
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        # create entropy coefficient
        self.log_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))

        # create optimizers
        self.q1_optimizer = Adam(self.q1.parameters(), lr=kwargs["value_learning_rate"])
        self.q2_optimizer = Adam(self.q2.parameters(), lr=kwargs["value_learning_rate"])
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=kwargs["policy_learning_rate"]
        )
        self.alpha_optimizer = Adam([self.log_alpha], lr=kwargs["alpha_learning_rate"])
        self.optimizer_dict = {
            "policy": self.policy_optimizer,
            "q1": self.q1_optimizer,
            "q2": self.q2_optimizer,
            "alpha": self.alpha_optimizer,
        }
        self.init_scheduler(**kwargs)


    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


class DSACT(AlgorithmBase):
    """DSAC algorithm with three refinements, higher performance and more stable.

    Paper: https://arxiv.org/abs/2310.05858

    :param float gamma: discount factor.
    :param float tau: param for soft update of target network.
    :param bool auto_alpha: whether to adjust temperature automatically.
    :param float alpha: initial temperature.
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
        self.delay_update = kwargs["delay_update"]
        self.mean_std1= None
        self.mean_std2= None
        self.tau_b = kwargs.get("tau_b", self.tau)
        self.rec1 = TimePerfRecorder("backward_q",print_interval=100)
        self.rec2 = TimePerfRecorder("backward_policy",print_interval=100)
        self.use_adapter = kwargs.get("policy_adapter_layers", None) is not None
        self._enable_adapter = False
        self.consider_last_act = kwargs.get("policy_consid_last_act", False)
        self.apapter_loss_weight = kwargs.get("policy_adapter_loss_weight", 0.01)
        self.max_perf_decrease = 0.05

    @property
    def adjustable_parameters(self):
        return (
            "gamma",
            "tau",
            "auto_alpha",
            "alpha",
            "delay_update",
        )

    def _local_update(self, data: DataDict, iteration: int) -> dict:
        tb_info = self._compute_gradient(data, iteration)
        self._update(iteration)
        return tb_info

    def get_remote_update_info(
        self, data: DataDict, iteration: int
    ) -> Tuple[dict, dict]:
        tb_info = self._compute_gradient(data, iteration)

        update_info = {
            "q1_grad": [p._grad for p in self.networks.q1.parameters()],
            "q2_grad": [p._grad for p in self.networks.q2.parameters()],
            "policy_grad": [p._grad for p in self.networks.policy.parameters()],
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

        for p, grad in zip(self.networks.q1.parameters(), q1_grad):
            p._grad = grad
        for p, grad in zip(self.networks.q2.parameters(), q2_grad):
            p._grad = grad
        for p, grad in zip(self.networks.policy.parameters(), policy_grad):
            p._grad = grad
        if self.auto_alpha:
            self.networks.log_alpha._grad = update_info["log_alpha_grad"]

        self._update(iteration)

    def _get_alpha(self, requires_grad: bool = False):
        if self.auto_alpha:
            alpha = self.networks.log_alpha.exp()
            if requires_grad:
                return alpha
            else:
                return alpha.item()
        else:
            return self.alpha

    def _compute_gradient(self, data: DataDict, iteration: int):
        start_time = time.time()
        if self.use_adapter and self.consider_last_act:
            last_act = data["act"] + torch.randn_like(data["act"]) * 0.2  # TODO: add to config 
            #NOTE: better to use real last act
            self.networks.policy.set_last_act(last_act)
            data.update({"last_act": last_act})
        obs = data["obs"]
        logits = self.networks.policy(obs)
        logits_mean, logits_std = torch.chunk(logits, chunks=2, dim=-1)
        policy_mean = torch.tanh(logits_mean).mean().item()
        policy_std = logits_std.mean().item()

        act_dist = self.networks.create_action_distributions(logits)
        new_act, new_log_prob = act_dist.rsample()
        new_act_mean = act_dist.mode()
        data.update({"new_act": new_act, "new_log_prob": new_log_prob, "new_act_mean": new_act_mean})

        self.networks.q1_optimizer.zero_grad()
        self.networks.q2_optimizer.zero_grad()
        loss_q, q1, q2, std1, std2, origin_q_loss = self._compute_loss_q(data)
        if loss_q.requires_grad:
            loss_q.backward() 


        with FreezeParameters([self.networks.q1, self.networks.q2]):

            self.networks.policy_optimizer.zero_grad()
            loss_policy, entropy = self._compute_loss_policy(data)
            loss_policy.backward()


        if self.auto_alpha:
            self.networks.alpha_optimizer.zero_grad()
            if self.networks.log_alpha.requires_grad:
                loss_alpha = self._compute_loss_alpha(data)
                loss_alpha.backward()

        tb_info = {
            "DSAC2/critic_avg_q1-RL iter": q1.mean().detach().item(),
            "DSAC2/critic_avg_q2-RL iter": q2.mean().detach().item(),
            "DSAC2/critic_avg_std1-RL iter": std1.mean().detach().item(),
            "DSAC2/critic_avg_std2-RL iter": std2.mean().detach().item(),
            "DSAC2/critic_avg_min_std1-RL iter": std1.min().detach().item(),
            "DSAC2/critic_avg_min_std2-RL iter": std2.min().detach().item(),
            "DSAC2/critic_avg_max_std1-RL iter": std1.max().detach().item(),
            "DSAC2/critic_avg_max_std2-RL iter": std2.max().detach().item(),
            tb_tags["loss_actor"]: loss_policy.item(),
            tb_tags["loss_critic"]: origin_q_loss.item(),
            "DSAC2/policy_mean-RL iter": policy_mean,
            "DSAC2/policy_std-RL iter": policy_std,
            "DSAC2/entropy-RL iter": entropy.item(),
            "DSAC2/alpha-RL iter": self._get_alpha(),
            "DSAC2/mean_std1": self.mean_std1,
            "DSAC2/mean_std2": self.mean_std2,
            tb_tags["alg_time"]: (time.time() - start_time) * 1000,
        }

        return tb_info

    def _q_evaluate(self, obs, act, qnet):
        StochaQ = qnet(obs, act)
        mean, std = StochaQ[..., 0], StochaQ[..., -1]
        normal = Normal(torch.zeros_like(mean), torch.ones_like(std))
        z = normal.sample()
        z = torch.clamp(z, -3, 3)
        q_value = mean + torch.mul(z, std)
        return mean, std, q_value

    def _compute_loss_q(self, data: DataDict):
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
        if self.use_adapter and self.consider_last_act:
            self.networks.policy_target.set_last_act(data["new_act"])
        logits_2 = self.networks.policy_target(noise_obs2)
        act2_dist = self.networks.create_action_distributions(logits_2)
        act2, log_prob_act2 = act2_dist.rsample()

        q1, q1_std, _ = self._q_evaluate(obs, act, self.networks.q1)
        _, q1_std_target, _ = self._q_evaluate(obs, act, self.networks.q1_target)
        q2, q2_std, _ = self._q_evaluate(obs, act, self.networks.q2)
        if self.mean_std1 is None:
            self.mean_std1 = torch.mean(q1_std.detach())
        else:
            self.mean_std1 = (1 - self.tau_b) * self.mean_std1 + self.tau_b * torch.mean(q1_std.detach())

        if self.mean_std2 is None:
            self.mean_std2 = torch.mean(q2_std.detach())
        else:
            self.mean_std2 = (1 - self.tau_b) * self.mean_std2 + self.tau_b * torch.mean(q2_std.detach())


        q1_next, _, q1_next_sample = self._q_evaluate(
            obs2, act2, self.networks.q1_target
        )
        
        q2_next, _, q2_next_sample = self._q_evaluate(
            obs2, act2, self.networks.q2_target
        )
        q_next = torch.min(q1_next, q2_next)
        q_next_sample = torch.where(q1_next < q2_next, q1_next_sample, q2_next_sample)

        target_q1, target_q1_bound = self._compute_target_q(
            rew,
            done,
            q1.detach(),
            self.mean_std1.detach(),
            q_next.detach(),
            q_next_sample.detach(),
            log_prob_act2.detach(),
        )
        
        target_q2, target_q2_bound = self._compute_target_q(
            rew,
            done,
            q2.detach(),
            self.mean_std2.detach(),
            q_next.detach(),
            q_next_sample.detach(),
            log_prob_act2.detach(),
        )

        q1_std_detach = torch.clamp(q1_std, min=0.).detach()
        q2_std_detach = torch.clamp(q2_std, min=0.).detach()
        bias = 0.1

        ratio1 = torch.pow(self.mean_std1, 2) / (torch.pow(q1_std_detach, 2) + bias).clamp(min=0.1, max=10)
        ratio2 = torch.pow(self.mean_std2, 2) / (torch.pow(q2_std_detach, 2) + bias).clamp(min=0.1, max=10)

        # q1_loss = torch.mean(ratio1 *(2*huber_loss(q1, target_q1, delta = 30, reduction='none')) + torch.log(q1_std +bias)
        #                               - q1_std * 2*huber_loss(q1.detach(), target_q1_bound, delta = 30, reduction='none')/(q1_std_detach +bias).pow(3)
        #                               )
                            
        # q2_loss = torch.mean(ratio2 *(2*huber_loss(q2, target_q2, delta = 30, reduction='none')) + torch.log(q2_std +bias)
        #                               - q2_std * 2*huber_loss(q2.detach(), target_q2_bound, delta = 30, reduction='none')/(q2_std_detach +bias).pow(3)
        #                              )
                            
        # form3
        q1_loss = torch.mean(ratio1 *(2*huber_loss(q1, target_q1, delta = 30, reduction='none')) 
                                      + q1_std *(q1_std_detach.pow(2) - 2*huber_loss(q1.detach(), target_q1_bound, delta = 30, reduction='none'))/(q1_std_detach.pow(3) +bias)
                            )
        q2_loss = torch.mean(ratio2 *(2*huber_loss(q2, target_q2, delta = 30, reduction='none'))
                                      + q2_std *(q2_std_detach.pow(2) - 2*huber_loss(q2.detach(), target_q2_bound, delta = 30, reduction='none'))/(q2_std_detach.pow(3) +bias)
                            )
        # form4 should similar to form 0
        # q1_loss = torch.mean(ratio1 *(2*huber_loss(q1, target_q1, delta = 30, reduction='none') 
        #                               + q1_std *(q1_std_detach.pow(2) - 2*huber_loss(q1.detach(), target_q1_bound, delta = 30, reduction='none'))/(q1_std_detach.pow(3) +bias)
        #                              )
        #                     )
        # q2_loss = torch.mean(ratio2 *(2*huber_loss(q2, target_q2, delta = 30, reduction='none')
        #                               + q2_std *(q2_std_detach.pow(2) - 2*huber_loss(q2.detach(), target_q2_bound, delta = 30, reduction='none'))/(q2_std_detach.pow(3) +bias)
        #                             )
        #                     )

        # fix 2
        # q1_loss = torch.mean(ratio1 *(2*huber_loss(q1, target_q1, delta = 30, reduction='none') 
        #                               + q1_std *(q1_std_detach.pow(2) - 2*huber_loss(q1.detach(), target_q1_bound, delta = 30, reduction='none'))/(q1_std_detach +bias)
        #                               )
        #                     )
        # q2_loss = torch.mean(ratio2 *(2*huber_loss(q2, target_q2, delta = 30, reduction='none')
        #                               + q2_std *(q2_std_detach.pow(2) - 2*huber_loss(q2.detach(), target_q2_bound, delta = 30, reduction='none'))/(q2_std_detach +bias)
        #                               )
        #                     )
        

        # q1_loss = torch.mean(ratio1 * ((q1 - target_q1).pow(2) + torch.log(q1_std +bias) -q1_std * (q1.detach() - target_q1_bound).pow(2) / (q1_std_detach + bias)))
        # q2_loss = torch.mean(ratio2 * ((q2 - target_q2).pow(2) + torch.log(q2_std +bias) -q2_std * (q2.detach() - target_q2_bound).pow(2) / (q2_std_detach + bias)))

        # q1_loss = (torch.pow(self.mean_std1, 2) + bias) * torch.mean(
        #     -(target_q1 - q1).detach() / ( torch.pow(q1_std_detach, 2)+ bias)*q1
        #     -((torch.pow(q1.detach() - target_q1_bound, 2)- q1_std_detach.pow(2) )/ (torch.pow(q1_std_detach, 3) +bias)
        #     )*q1_std
        # )

        # q2_loss = (torch.pow(self.mean_std2, 2) + bias)*torch.mean(
        #     -(target_q2 - q2).detach() / ( torch.pow(q2_std_detach, 2)+ bias)*q2
        #     -((torch.pow(q2.detach() - target_q2_bound, 2)- q2_std_detach.pow(2) )/ (torch.pow(q2_std_detach, 3) +bias)
        #     )*q2_std
        # )
        with torch.no_grad():
            # origin_q_loss = 0.5 * (q1_loss + q2_loss).detach()
            # form 0
            origin_q1_loss = torch.mean(ratio1 *(2*huber_loss(q1, target_q1, delta = 30, reduction='none') + torch.log(q1_std +bias)
                                      + 0.5 * 2*huber_loss(q1.detach(), target_q1_bound, delta = 30, reduction='none')/(q1_std + bias).pow(2)
                                      )
                            )
            
            origin_q2_loss = torch.mean(ratio2 *(2*huber_loss(q2, target_q2, delta = 30, reduction='none') + torch.log(q2_std +bias)
                                      + 0.5 * 2*huber_loss(q2.detach(), target_q2_bound, delta = 30, reduction='none')/(q2_std + bias).pow(2)
                                      )
                            )   
            origin_q_loss = origin_q1_loss + origin_q2_loss
        


        return q1_loss +q2_loss, q1, q2, q1_std, q2_std, origin_q_loss

    def _compute_target_q(self, r, done, q,q_std, q_next, q_next_sample, log_prob_a_next):
        target_q = r + (1 - done) * self.gamma * (
            q_next - self._get_alpha() * log_prob_a_next
        )
        target_q_sample = r + (1 - done) * self.gamma * (
            q_next_sample - self._get_alpha() * log_prob_a_next
        )
        td_bound = 3 * q_std
        difference = torch.clamp(target_q_sample - q, -td_bound, td_bound)
        target_q_bound = q + difference
        return target_q.detach(), target_q_bound.detach()

    def _compute_loss_policy(self, data: DataDict):
        obs, new_act, new_log_prob = data["obs"], data["new_act"], data["new_log_prob"]
        obs = data.get("raw_obs", obs)
        q1, _, _ = self._q_evaluate(obs, new_act, self.networks.q1)
        q2, _, _ = self._q_evaluate(obs, new_act, self.networks.q2)
        loss_policy = (self._get_alpha() * new_log_prob - torch.min(q1,q2)).mean()
        if self.use_adapter and self._enable_adapter:
            q = torch.min(q1, q2)
            q1_freeze, _, _ = self._q_evaluate(obs, new_act, self.freeze_q1)
            q2_freeze, _, _ = self._q_evaluate(obs, new_act, self.freeze_q2)
            q_freeze = torch.min(q1_freeze, q2_freeze)
            adapter_loss = self._compute_loss_adapter(data, q_freeze, q)
            loss_policy += adapter_loss

        entropy = -new_log_prob.detach().mean()
        return loss_policy, entropy
    def _compute_loss_adapter(self, data: DataDict, q_freeze, q):
        adapter_loss = 0
        last_act = data["last_act"]
        if self.consider_last_act:
            self.networks.policy.set_last_act(last_act)
            
        obs = data["obs"]
        obs = data.get("raw_obs", obs)
        rel_q_decrease = (q_freeze - q) / q_freeze.abs().clamp(min=1e-6).detach()
        rel_loss_weight = self.apapter_loss_weight*(1 - torch.cos(torch.pi/2*(1- (rel_q_decrease/self.max_perf_decrease).clamp(min=0, max=1)))).unsqueeze(-1)
        if self.consider_last_act:
            new_act_mean = data["new_act_mean"]
            self.networks.policy.set_last_act(last_act)
            temporal_smooth_loss = torch.mean(rel_loss_weight*(new_act_mean - last_act).pow(2))
            adapter_loss += temporal_smooth_loss
        delta = 1e-7 + torch.randn_like(obs) * 1e-6  # B, 2D
        delta_norm = torch.norm(delta, dim=-1, p=2)
        noised_obs = obs + delta
        logits = self.networks.policy(noised_obs)
        noised_act_dist = self.networks.create_action_distributions(logits)
        noised_act_mean = noised_act_dist.mode()
        #print('norm_mean',norm.mean())
        #print('norm_max',norm.max())

        # jacobi = (out2 - out1) / delta
        # norm = torch.norm(jacobi[:,0,:self.act_dim], dim=-1, p=1).mean()
        norm= (torch.norm((new_act_mean - noised_act_mean), dim=-1, p=1)/delta_norm)
        spatial_smooth_loss = torch.mean(rel_loss_weight*norm)
        adapter_loss += spatial_smooth_loss
        return adapter_loss
        

    def _compute_loss_alpha(self, data: DataDict):
        new_log_prob = data["new_log_prob"]
        loss_alpha = (
            -self.networks.log_alpha
            * (new_log_prob.detach() + self.target_entropy).mean()
        )
        return loss_alpha
    

    def _update(self, iteration: int):
        self.networks.q1_optimizer.step()
        self.networks.q2_optimizer.step()

        if iteration % self.delay_update == 0:
            self.networks.policy_optimizer.step()

            if self.auto_alpha:
                self.networks.alpha_optimizer.step()

            with torch.no_grad():
                polyak = 1 - self.tau
                for p, p_targ in zip(
                    self.networks.q1.parameters(), self.networks.q1_target.parameters()
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
                for p, p_targ in zip(
                    self.networks.q2.parameters(), self.networks.q2_target.parameters()
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
                for p, p_targ in zip(
                    self.networks.policy.parameters(),
                    self.networks.policy_target.parameters(),
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def change_mode(self):
        if self.use_adapter:
            self._enable_adapter = True
            self.networks.policy.enable_adapter()
            self.freeze_q1 = deepcopy(self.networks.q1)
            self.freeze_q2 = deepcopy(self.networks.q2)
            for p in self.freeze_q1.parameters():
                p.requires_grad = False
            for p in self.freeze_q2.parameters():
                p.requires_grad = False