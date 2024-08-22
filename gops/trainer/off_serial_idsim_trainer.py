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
#  Update Date: 2023-11-24, Guojian Zhan: create for idsim env

__all__ = ["OffSerialIdsimTrainer"]

from cmath import inf
import os
import time

import ray
import torch

from typing import Dict
from torch.utils.tensorboard import SummaryWriter

from gops.utils.common_utils import ModuleOnDevice
from gops.utils.parallel_task_manager import TaskPool
from gops.utils.tensorboard_setup import add_scalars, tb_tags
from gops.utils.log_data import LogData
from gops.trainer.off_serial_trainer import OffSerialTrainer
from gops.trainer.idsim_train_evaluator import idsim_tb_tags_dict

class OffSerialIdsimTrainer(OffSerialTrainer):
    def __init__(self, alg, sampler, buffer, evaluator, **kwargs):
        self.alg = alg
        self.sampler = sampler
        self.buffer = buffer
        self.per_flag = kwargs["buffer_name"].startswith("prioritized") # FIXME: hard code
        self.evaluator = evaluator

        # create center network
        self.networks = self.alg.networks
        self.networks.eval()

        # initialize center network
        if kwargs["ini_network_dir"] is not None:
            self.networks.load_state_dict(torch.load(kwargs["ini_network_dir"]))
        if kwargs["pi_ini_network_dir"] is not None:
            model_para_dict = self.networks.state_dict()
            pi_para_dict = {k: v for k, v in model_para_dict.items() if "pi_net" in k}
            network_dict = torch.load(kwargs["pi_ini_network_dir"])
            for key, _ in pi_para_dict.items():  # replace the pi network parameters
                pi_para_dict[key] = network_dict[key]
            self.networks.load_state_dict(model_para_dict)  # load the model parameters
            self.sampler.load_state_dict.remote(model_para_dict)  # load the model parameters
            self.evaluator.load_state_dict.remote(model_para_dict)

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

        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
        # flush tensorboard at the beginning
        add_scalars(
            {tb_tags["alg_time"]: 0, tb_tags["sampler_time"]: 0}, self.writer, 0
        )
        self.writer.flush()

        # buffer statistics
        head = "Iter, Mean, Std, 0%, 25%, 50%, 75%, 100%\n"
        with open(self.save_folder + "/buffer_vx_stat.csv", "w") as f:
            f.write(head)
        with open(self.save_folder + "/buffer_y_ref_stat.csv", "w") as f:
            f.write(head)



        self.sampler_tb_dict = LogData()

        # create evaluation tasks
        self.evluate_tasks = TaskPool()
        self.last_eval_iteration = 0

        # create sampler tasks
        self.sampler_tasks = TaskPool()
        self.last_sampler_network_update_iteration = 0
        self.sampler_network_update_interval = kwargs.get("sampler_network_update_interval", 100)
        self.last_sampler_save_iteration = 0

        self.use_gpu = kwargs["use_gpu"]
        if self.use_gpu:
            self.networks.cuda()


        # pre sampling
        while self.buffer.size < kwargs["buffer_warm_size"]:
            samples, _ = ray.get(self.sampler.sample.remote())
            self.buffer.add_batch(samples)

        self.start_time = time.time()

    def step(self):
        # trainning phrase
        if self.iteration == self.freeze_iteration:
            self.change_train_phase()
        # sampling
        if self.iteration % self.sample_interval == 0:
            if self.sampler_tasks.count == 0:
                # There is no sampling task, add one.
                self._add_sample_task()

        # replay
        replay_samples = self.buffer.sample_batch(self.replay_batch_size)

        # learning
        for k, v in replay_samples.items():
            replay_samples[k] = v.to(self.networks.device)

        self.networks.train()
        if self.per_flag:
            alg_tb_dict, idx, new_priority = self.alg.local_update(
                replay_samples, self.iteration
            )
            self.buffer.update_batch(idx, new_priority)
        else:
            alg_tb_dict = self.alg.local_update(replay_samples, self.iteration)
        self.networks.eval()


        # sampling
        if self.iteration % self.sample_interval == 0:
            while self.sampler_tasks.completed_num == 0:
                # There is no completed sampling task, wait.
                time.sleep(0.001)
            # Sampling task is completed, get samples and add another one.
            objID = next(self.sampler_tasks.completed())[1]
            sampler_samples, sampler_tb_dict = ray.get(objID)
            self._add_sample_task()
            self.buffer.add_batch(sampler_samples)
            if (self.iteration - self.last_sampler_save_iteration) >= self.log_save_interval:
                self.sampler_tb_dict.add_average(sampler_tb_dict)

        # log
        if self.iteration % self.log_save_interval == 0:
            print("Iter = ", self.iteration)
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)
            add_scalars(self.sampler_tb_dict.pop(), self.writer, step=self.iteration)
            
        if self.iteration % (self.max_iteration//100) == 0:  # TODO: Hard code
            pass
            # stat_dict: Dict = self.buffer.sample_statistic(self.iteration)
            # with open(self.save_folder + "/buffer_vx_stat.csv", "a") as f:
            #     f.write(stat_dict["vx"] + "\n")
            # with open(self.save_folder + "/buffer_y_ref_stat.csv", "a") as f:
            #     f.write(stat_dict["y_ref"] + "\n")
        if self.per_flag and self.iteration % (self.max_iteration//20) == 0:  # TODO: Hard code
            self.buffer.save_data_dist(self.save_folder, self.iteration, replay_samples)

        # save
        if self.iteration % self.apprfunc_save_interval == 0:
            self.save_apprfunc()

        # evaluate
        if self.iteration - self.last_eval_iteration >= self.eval_interval:
            if self.evluate_tasks.count == 0:
                # There is no evaluation task, add one.
                self._add_eval_task()
            elif self.evluate_tasks.completed_num == 1:
                # Evaluation tasks is completed, log data and add another one.
                objID = next(self.evluate_tasks.completed())[1]
                avg_tb_eval_dict = ray.get(objID)
                total_avg_return = avg_tb_eval_dict['total_avg_return']
                self._add_eval_task()

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
                        + "/apprfunc/apprfunc_{}_opt.pkl".format(self.iteration),
                    )

                self.writer.add_scalar(
                    tb_tags["Buffer RAM of RL iteration"],
                    self.buffer.__get_RAM__(),
                    self.iteration,
                )
                self.writer.add_scalar(
                    tb_tags["TAR of RL iteration"], total_avg_return, self.iteration
                )
                self.writer.add_scalar(
                    tb_tags["TAR of replay samples"],
                    total_avg_return,
                    self.iteration * self.replay_batch_size,
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
                for key, value in avg_tb_eval_dict.items():
                    if key != "total_avg_return":
                        self.writer.add_scalar(idsim_tb_tags_dict[key], value, self.iteration)

    def train(self):
        while self.iteration < self.max_iteration:
            self.step()
            self.iteration += 1

        self.save_apprfunc()
        self.writer.flush()

    def save_apprfunc(self):
        torch.save(
            self.networks.state_dict(),
            self.save_folder + "/apprfunc/apprfunc_{}.pkl".format(self.iteration),
        )

    def _add_eval_task(self):
        self.evaluator.load_state_dict.remote(self.networks.state_dict())
        self.evluate_tasks.add(
            self.evaluator,
            self.evaluator.run_evaluation.remote(self.iteration)
        )
        self.last_eval_iteration = self.iteration
    def change_train_phase(self):
        #self.networks.policy.freeze()
        #self.networks.log_alpha.requires_grad = False
        # self.networks.q1.freeze()
        # self.networks.q2.freeze()
        self.buffer.change_mode()
        self.sampler.change_mode.remote()
        
