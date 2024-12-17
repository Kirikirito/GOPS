#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Serial trainer for off-policy RL algorithms
#  Update Date: 2021-05-21, Shengbo LI: Format Revise
#  Update Date: 2022-04-14, Jiaxin Gao: decrease parameters copy times
#  Update: 2022-12-05, Wenhan Cao: add annotation

__all__ = ["OffSerialTrainer"]

from cmath import inf
import os
import time
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

from gops.utils.common_utils import ModuleOnDevice
import multiprocessing
import threading
import io
from gops.utils.parallel_task_manager import TaskPool
from gops.utils.tensorboard_setup import add_scalars, tb_tags

class OffSerialTrainer:
    def __init__(self, alg, sampler, buffer, evaluator, **kwargs):
        self.alg = alg
        self.sampler = sampler
        self.buffer = buffer
        self.per_flag = kwargs["buffer_name"] == "prioritized_replay_buffer"
        self.evaluator = evaluator

        # create center network
        self.networks = self.alg.networks
        # create evaluation tasks
        self.evluate_tasks = TaskPool()
        self.last_eval_iteration = 0

        # create sampler tasks
        self.sampler_tasks = TaskPool()
        self.last_sampler_network_update_iteration = 0
        self.sampler_network_update_interval = kwargs.get("sampler_network_update_interval", 100)
        self.last_sampler_save_iteration = 0
        self.networks.eval()
        
        self.use_gpu = kwargs["use_gpu"]
        if self.use_gpu:
            self.networks.cuda()

        # initialize center network
        if kwargs["ini_network_dir"] is not None:
            self.networks.load_state_dict(torch.load(kwargs["ini_network_dir"]))
        self.ini_buffer = kwargs.get("ini_buffer", None)
        if self.ini_buffer is not None:
            self.buffer.load_hd5(kwargs["ini_buffer"])
            if hasattr(self.alg.networks, "load_optimizer"):
                try:
                    self.alg.networks.load_optimizer(kwargs["ini_buffer"])
                except:
                    print("Optimizer is not loaded")
                    pass
                
            print("Buffer is loaded from ", kwargs["ini_buffer"])

        self.replay_batch_size = kwargs["replay_batch_size"]
        self.max_iteration = kwargs["max_iteration"]
        self.freeze_iteration = kwargs.get("freeze_iteration", self.max_iteration +1)
        self.sample_interval = kwargs.get("sample_interval", 1)
        self.log_save_interval = kwargs["log_save_interval"]
        self.apprfunc_save_interval = kwargs["apprfunc_save_interval"]
        self.eval_interval = kwargs["eval_interval"]
        self.best_tar = -inf
        self.save_folder = kwargs["save_folder"]
        self.iteration = 0
        self.use_adapter = kwargs.get("policy_adapter_layers", None) is not None
        self.freeze_q = kwargs.get("freeze_q", False)
        self.freeze_policy = kwargs.get("freeze_policy", False)
        self.enable_seq = kwargs.get("enable_seq", False)
        if self.enable_seq:
            self.buffer.enable_seq()
            self.sampler.change_mode.remote()
            self.evaluator.change_mode.remote()
        
        # save buffer
        self.save_buffer = kwargs.get("save_buffer", False)


        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
        # flush tensorboard at the beginning
        add_scalars(
            {tb_tags["alg_time"]: 0, tb_tags["sampler_time"]: 0}, self.writer, 0
        )
        self.writer.flush()

        # pre sampling
        while self.buffer.size < kwargs["buffer_warm_size"] and not self.ini_buffer:
            samples, _ = ray.get(self.sampler.sample.remote())
            self.buffer.add_batch(samples)

        self.start_time = time.time()
        self.sampler_samples = None
        self.sampler_tb_dict = None
        self.total_time = 0
        
    def _replay(self):
        replay_samples = self.buffer.sample_batch(self.replay_batch_size)
        if self.use_gpu:
            for k, v in replay_samples.items():
                replay_samples[k] = v.to(self.networks.device, non_blocking=True)
            torch.cuda.synchronize()
        else:
            for k, v in replay_samples.items():
                replay_samples[k] = v.to(self.networks.device)
        return replay_samples

    def step(self):
        # trainning phrase
        if self.iteration == self.freeze_iteration:
            self.change_train_phase()

        step_start_time = time.time()

        # # update sampler networks
        # if self.iteration % 100 == 0:
        #     cpu_state_dict = {k: v.cpu() for k, v in self.networks.state_dict().items()}
        #     ray.get(self.sampler.load_state_dict.remote(cpu_state_dict))

        # sampling
        if self.iteration % self.sample_interval == 0:
            with torch.no_grad():
                if self.sampler_tasks.count == 0:
                    # There is no sampling task, add one.
                    self._add_sample_task()
        
        # replay
        replay_samples = self._replay()

        if self.per_flag:
            alg_tb_dict, idx, new_priority = self.alg.local_update(
                replay_samples, self.iteration
            )
            self.buffer.update_batch(idx, new_priority)
        else:
            alg_tb_dict = self.alg.local_update(replay_samples, self.iteration)

        # sampling
        if self.iteration % self.sample_interval == 0:
            with torch.no_grad():
                while self.sampler_tasks.completed_num == 0:
                    # There is no completed sampling task, wait.
                    time.sleep(0.001)
                # Sampling task is completed, get samples and add another one.
                objID = next(self.sampler_tasks.completed())[1]
                sampler_samples, sampler_tb_dict = ray.get(objID)
                self._add_sample_task()
                self.buffer.add_batch(sampler_samples)
                if (self.iteration - self.last_sampler_save_iteration) >= self.log_save_interval:
                    add_scalars(sampler_tb_dict, self.writer, step=self.iteration)
                    self.last_sampler_save_iteration = self.iteration

        # log
        if self.iteration % self.log_save_interval == 0:
            print("Iter = ", self.iteration)
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)

        # evaluate
        if self.iteration - self.last_eval_iteration >= self.eval_interval or self.iteration == 0 or self.iteration == self.max_iteration:
            if self.evluate_tasks.count == 0:
                # There is no evaluation task, add one.
                self._add_eval_task()
            elif self.evluate_tasks.completed_num == 1:
                # Evaluation tasks is completed, log data and add another one.
                self.record_eval()
                self._add_eval_task()
            
            if self.iteration == self.max_iteration:
                if self.evluate_tasks.count != 0:
                    # Evaluation tasks is not completed, wait.
                    print("Waiting for evaluation task to complete ...")
                    while self.evluate_tasks.completed_num == 0:
                        time.sleep(0.001)
                    self.record_eval()
                
                if self.last_eval_iteration < self.max_iteration:
                    print("Adding evaluation task for the last iteration ...")
                    self._add_eval_task()
                    while self.evluate_tasks.completed_num == 0:
                        time.sleep(0.001)
                    print("Evaluation task for the last iteration is completed.")
                    self.record_eval()
                    print("Evaluation is finished!")

        # save
        if self.iteration % self.apprfunc_save_interval == 0:
            self.save_apprfunc()

    
    def record_eval(self, ):
        eval_ite = self.last_eval_iteration
        objID = next(self.evluate_tasks.completed())[1]
        eval_result = ray.get(objID)
        total_avg_return = eval_result["episode_return"]

        if (
            total_avg_return >= self.best_tar
            and self.iteration >= self.max_iteration / 5
        ):
            self.best_tar = total_avg_return
            print("Best return = {}!".format(str(self.best_tar)))

            for filename in os.listdir(self.save_folder + "/apprfunc/"):
                if filename.endswith("_opt.pkl"):
                    os.remove(self.save_folder + "/apprfunc/" + filename)

            torch.save(
                self.networks.state_dict(),
                self.save_folder
                + "/apprfunc/apprfunc_{}_opt.pkl".format(eval_ite),
            )

        self.writer.add_scalar(
            tb_tags["Buffer RAM of RL iteration"],
            self.buffer.__get_RAM__(),
            eval_ite,
        )
        self.writer.add_scalar(
            tb_tags["TAR of RL iteration"], total_avg_return, eval_ite
        )
        self.writer.add_scalar(
            tb_tags["TAR of replay samples"],
            total_avg_return,
            eval_ite * self.replay_batch_size,
        )
        self.writer.add_scalar(
            tb_tags["TAR of total time"],
            total_avg_return,
            int(time.time() - self.start_time),
        )
        self.writer.add_scalar(
            tb_tags["TAR of collected samples"],
            total_avg_return,
            ray.get(self.sampler.get_total_sample_number.remote()),
        )
        
        # Smoothness
        if "afr" in eval_result.keys():  # FIXME: afr is not always calculated
            self.writer.add_scalar(
                tb_tags["afr"], eval_result["afr"], eval_ite
            )
            self.writer.add_scalar(
                tb_tags["afr_std"], eval_result["afr_std"], eval_ite
            )
            self.writer.add_scalar(
                tb_tags["sdv"], eval_result["sdv"], eval_ite
            )
            # mean weight frequency
            self.writer.add_scalar(
                tb_tags["mwf"], eval_result["mwf"], eval_ite
            )
            
            
    def train(self):
        while self.iteration <= self.max_iteration:
            start_time = time.time()
            self.step()
            end_time = time.time()
            self.total_time += end_time - start_time
            self.iteration += 1
            # print("avg time per iteration: ", self.total_time / self.iteration)
        print("Training is finished!")
        self.iteration = self.max_iteration
        self.save_apprfunc()

        self.writer.flush()

    def save_apprfunc(self):
        torch.save(
            self.networks.state_dict(),
            self.save_folder + "/apprfunc/apprfunc_{}.pkl".format(self.iteration),
        )
        self.sampler.save.remote(self.save_folder + "/sampler")
        

    def _add_eval_task(self):
        with torch.no_grad():
            # with ModuleOnDevice(self.networks, "cpu"):
            self.evaluator.load_state_dict.remote(self.networks.state_dict())
            self.evluate_tasks.add(
                self.evaluator,
                self.evaluator.run_evaluation.remote(self.iteration)
            )
        self.last_eval_iteration = self.iteration    

    def _add_sample_task(self):
        with torch.no_grad():
            if (self.iteration - self.last_sampler_network_update_iteration) >= self.sampler_network_update_interval:
                self.last_sampler_network_update_iteration = self.iteration
                ret = self.sampler.load_state_dict.remote(self.networks.state_dict())
                ret = ray.get(ret)
                # wait for the sampler to load the network
                
            self.sampler_tasks.add(
                self.sampler,
                self.sampler.sample.remote()
            )
    def change_train_phase(self):
        self.save_apprfunc()
        if self.save_buffer:
            buffer_dir = "/root/autodl-tmp/thesis" + self.save_folder
            if not os.path.exists(buffer_dir):
                os.makedirs(buffer_dir)
            self.buffer.save_hd5(buffer_dir)
            if hasattr(self.alg.networks, "save_optimizer"):
                self.alg.networks.save_optimizer(buffer_dir)
                
        if self.freeze_policy:
            self.networks.policy.freeze()
        #self.networks.log_alpha.requires_grad = False
        # self.networks.q1.freeze()
        # self.networks.q2.freeze()
        if self.freeze_q:
            if hasattr(self.networks, "q1"):
                self.networks.q1.freeze()
                self.networks.q2.freeze()
            if hasattr(self.networks, "q"):
                self.networks.q.freeze()
        self.buffer.change_mode()
        self.sampler.change_mode.remote()
        self.evaluator.change_mode.remote()
        if self.use_adapter:
            self.networks.policy.enable_adapter()
            self.alg.change_mode()

        
