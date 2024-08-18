#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: example for dsac + pendulum + mlp + offserial
#  Update Date: 2021-03-05, Gu Ziqing: create example

import argparse
import os
import numpy as np
import json

from copy import deepcopy

from gops.create_pkg.create_alg import create_alg
from gops.create_pkg.create_buffer import create_buffer
from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_evaluator import create_evaluator
from gops.create_pkg.create_sampler import create_sampler
from gops.create_pkg.create_trainer import create_trainer
from gops.utils.init_args import init_args
from gops.utils.plot_evaluation import plot_all
from gops.utils.tensorboard_setup import start_tensorboard, save_tb_to_csv
from gops.env.env_gen_ocp.resources.idsim_config_ml import get_idsim_env_config, get_idsim_model_config, pre_horizon, cal_idsim_obs_scale, cal_idsim_pi_paras

os.environ["OMP_NUM_THREADS"] = "2"
os.environ['RAY_memory_monitor_refresh_ms'] = "0"  # disable memory monitor
if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for users
    parser.add_argument("--env_id", type=str, default="pyth_idsim", help="id of environment")
    parser.add_argument("--env_scenario", type=str, default="multilane", help="crossroad / multilane")
    parser.add_argument("--num_threads_main", type=int, default=4, help="Number of threads in main process")
    env_scenario = parser.parse_known_args()[0].env_scenario

    base_env_config = get_idsim_env_config(env_scenario)
    base_env_model_config = get_idsim_model_config(env_scenario)
    parser.add_argument("--extra_env_config", type=str, default=r'{}')
    parser.add_argument("--extra_env_model_config", type=str, default=r'{}')
    extra_env_config = parser.parse_known_args()[0].extra_env_config
    print(extra_env_config )
    extra_env_config = json.loads(extra_env_config)
    extra_env_model_config = parser.parse_known_args()[0].extra_env_model_config
    extra_env_model_config = json.loads(extra_env_model_config)
    base_env_config.update(extra_env_config)
    base_env_model_config.update(extra_env_model_config)
    parser.add_argument("--env_config", type=dict, default=base_env_config)
    parser.add_argument("--env_model_config", type=dict, default=base_env_model_config)
    parser.add_argument("--scenerios_list", type=list, default=[':22',':22','22:'])

    # parser.add_argument("--vector_env_num", type=int, default=10, help="Number of vector envs")
    # parser.add_argument("--vector_env_type", type=str, default='async', help="Options: sync/async")
    # parser.add_argument("--gym2gymnasium", type=bool, default=True, help="Convert Gym-style env to Gymnasium-style")

    parser.add_argument("--ego_scale", type=list, default=[1, 20, 20, 1, 4, 1, 4] ) #  vx, vy, r, last_last_acc, last_last_steer, last_acc, last_steer
    parser.add_argument("--sur_scale", type=list, default=[0.2, 1, 1, 10, 1, 1, 1, 1] ) #  rel_x, rel_y , cos(phi), sin(phi), speed, length, width, mask
    parser.add_argument("--ref_scale", type=list, default=[0.2, 1, 1, 10, 1] ) # ref_x ref_y ref_cos(ref_phi) ref_sin(ref_phi), error_v
    ego_scale = parser.parse_known_args()[0].ego_scale
    sur_scale = parser.parse_known_args()[0].sur_scale
    ref_scale = parser.parse_known_args()[0].ref_scale
    obs_scale = cal_idsim_obs_scale(
        ego_scale=ego_scale,
        sur_scale=sur_scale,
        ref_scale=ref_scale,
        env_config=base_env_config,
        env_model_config=base_env_model_config
    )
    parser.add_argument("--obs_scale", type=dict, default=obs_scale)
    parser.add_argument("--repeat_num", type=int, default=4, help="action repeat num")

    parser.add_argument("--algorithm", type=str, default="DSACTPI", help="RL algorithm")
    parser.add_argument("--enable_cuda", default=False, help="Enable CUDA")
    parser.add_argument("--seed", default=1, help="seed")

    ################################################
    # 1. Parameters for environment
    parser.add_argument("--reward_scale", type=float, default=1, help="reward scale factor")
    parser.add_argument("--action_type", type=str, default="continu", help="Options: continu/discret")
    parser.add_argument("--is_render", type=bool, default=False, help="Draw environment animation")
    parser.add_argument("--is_adversary", type=bool, default=False, help="Adversary training")
    parser.add_argument("--is_constrained", type=bool, default=False, help="Adversary training")

    ################################################
    # 2.1 Parameters of value approximate function
    parser.add_argument(
        "--value_func_name",
        type=str,
        default="ActionValueDistri",
        help="Options: StateValue/ActionValue/ActionValueDis/ActionValueDistri",
    )
    parser.add_argument("--value_func_type", type=str, default="PINet", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS")
    parser.add_argument("--value_hidden_sizes", type=list, default=[256, 256,256])
    parser.add_argument("--value_std_type", type=str, default='mlp_separated', help="Options: mlp_separated/mlp_shared")
    parser.add_argument(
        "--value_hidden_activation", type=str, default="gelu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--value_output_activation", type=str, default="linear", help="Options: linear/tanh")


    # 2.2 Parameters of policy approximate function
    parser.add_argument(
        "--policy_func_name",
        type=str,
        default="StochaPolicy",
        help="Options: None/DetermPolicy/FiniteHorizonPolicy/StochaPolicy",
    )
    parser.add_argument(
        "--policy_func_type", type=str, default="PINet", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS"
    )
    parser.add_argument(
        "--policy_act_distribution",
        type=str,
        default="TanhGaussDistribution",
        help="Options: default/TanhGaussDistribution/GaussDistribution",
    )
    parser.add_argument("--policy_hidden_sizes", type=list, default=[256,256,256])
    parser.add_argument(
        "--policy_hidden_activation", type=str, default="gelu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--policy_output_activation", type=str, default="linear", help="Options: linear/tanh")
    parser.add_argument("--policy_min_log_std", type=int, default=-20)
    parser.add_argument("--policy_max_log_std", type=int, default=0.5)

    # 2.3 Parameters of shared approximate function
    pi_paras = cal_idsim_pi_paras(env_config=base_env_config, env_model_config=base_env_model_config)
    parser.add_argument("--target_PI", type=bool, default=True)
    parser.add_argument("--enable_self_attention", type=bool, default=True)
    parser.add_argument("--pi_begin", type=int, default=pi_paras["pi_begin"])
    parser.add_argument("--pi_end", type=int, default=pi_paras["pi_end"])
    parser.add_argument("--enable_mask", type=bool, default=True)
    parser.add_argument("--obj_dim", type=int, default=pi_paras["obj_dim"])
    parser.add_argument("--head_num", type=int, default=8)
    parser.add_argument("--pi_out_dim", type=int, default= 256)
    parser.add_argument("--pi_hidden_sizes", type=list, default=[256,256])
    parser.add_argument("--pi_hidden_activation", type=str, default="gelu")
    parser.add_argument("--pi_output_activation", type=str, default="gelu")
    parser.add_argument("--freeze_pi_net", type=str, default="none")
    parser.add_argument("--encoding_others", type=bool, default=False)
    parser.add_argument("--others_hidden_sizes", type=list, default=[64,64])
    parser.add_argument("--others_hidden_activation", type=str, default="gelu")
    parser.add_argument("--others_output_activation", type=str, default="linear")
    parser.add_argument("--others_out_dim", type=int, default=32)
    max_iter = 1_000_000
    parser.add_argument("--policy_scheduler", type=json.loads, default={
        "name": "CosineAnnealingLR",
        "params": {
                "T_max": max_iter,
            }
    })

    parser.add_argument("--q1_scheduler", type=json.loads, default={
        "name": "CosineAnnealingLR",
        "params": {
                "T_max": max_iter,
            }
    })
    parser.add_argument("--q2_scheduler", type=json.loads, default={
        "name": "CosineAnnealingLR",
        "params": {
                "T_max": max_iter,
            }
    })
    parser.add_argument("--pi_scheduler", type=json.loads, default={
        "name": "CosineAnnealingLR",
        "params": {
                "T_max": max_iter,
            }
    })

    parser.add_argument("--alpha_scheduler", type=json.loads, default={
        "name": "CosineAnnealingLR",
        "params": {
                "T_max": max_iter,
            }
    })
    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--value_learning_rate", type=float, default=1e-4)
    parser.add_argument("--policy_learning_rate", type=float, default=1e-4)
    parser.add_argument("--pi_learning_rate", type=float, default=1e-5)
    parser.add_argument("--alpha_learning_rate", type=float, default=3e-4)

    # special parameter
    parser.add_argument("--gamma", type=float, default=0.90)
    parser.add_argument("--tau", type=float, default=0.001)
    parser.add_argument("--auto_alpha", type=bool, default=True)
    parser.add_argument("--alpha", type=bool, default=0.2)
    parser.add_argument("--delay_update", type=int, default=2)
    parser.add_argument("--TD_bound", type=float, default=10)
    parser.add_argument("--bound", default=True)

    ################################################
    # 4. Parameters for trainer
    parser.add_argument(
        "--trainer",
        type=str,
        default="off_serial_idcpu_trainer",
        help="Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer",
    )
    # Maximum iteration number
    parser.add_argument("--max_iteration", type=int, default=max_iter)
    parser.add_argument(
        "--ini_network_dir",
        type=str,
        default=None
    )
    trainer_type = parser.parse_known_args()[0].trainer
    # 4.1. Parameters for off_serial_trainer
    parser.add_argument(
        "--buffer_name", type=str, default="prioritized_stratified_replay_buffer", help="Options:replay_buffer/prioritized_replay_buffer"
    )
    parser.add_argument(
        "--category_num", type=int, default=6, help="Number of categories for stratified replay buffer")
    # Size of collected samples before training
    parser.add_argument("--buffer_warm_size", type=int, default=10000)
    # Max size of reply buffer
    parser.add_argument("--buffer_max_size", type=int, default=225000)
    # Batch size of replay samples from buffer
    parser.add_argument("--replay_batch_size", type=int, default=256)
    # Period of sampling
    parser.add_argument("--sample_interval", type=int, default=20)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="off_sampler", help="Options: on_sampler/off_sampler")
    # Batch size of sampler for buffer store
    parser.add_argument("--sample_batch_size", type=int, default=80)
    # Add noise to action for better exploration
    parser.add_argument("--noise_params", type=dict, default={"mean": np.array([0,0], dtype=np.float32), "std": np.array([0.2,0.2], dtype=np.float32),},
        help="used for continuous action space")

    ################################################
    # 6. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="idsim_train_evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--eval_save", type=str, default=False, help="save evaluation data")

    ################################################
    # 7. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    # Save value/policy every N updates
    parser.add_argument("--apprfunc_save_interval", type=int, default=50000)
    # Save key info every N updates
    parser.add_argument("--log_save_interval", type=int, default=1000)

    ################################################
    eval_env_config = {
        "use_multiple_path_for_multilane": False,
        "takeover_bias": False,
        "scenario_reuse": 1,
        "warmup_time": 100.0,
        "takeover_bias_x": (0.0, 1),
        "takeover_bias_y": (0.0, 1),
        "takeover_bias_phi": (0.0, 0.02),
    }
    # Get parameter dictionary
    args = vars(parser.parse_args())
    args["eval_env_config"] = eval_env_config
    env = create_env(**{**args, "vector_env_num": None})
    args = init_args(env, **args)
    # env.close()

    # start_tensorboarsd(args["save_folder"])
    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)
    # Step 2: create sampler in trainer
    sampler = create_sampler(**args)
    # Step 3: create buffer in trainer
    buffer = create_buffer(**args)
    # Step 4: create evaluator in trainer
    eval_args = deepcopy(args)
    eval_args["env_config"].update(eval_env_config)
    eval_args["repeat_num"] = None  
    evaluator = create_evaluator(**eval_args)
    # Step 5: create trainer
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)

    ################################################
    # Start training ... ...
    trainer.train()
    print("Training is finished!")

    ################################################
    # Plot and save training figures
    # plot_all(args["save_folder"])
    # save_tb_to_csv(args["save_folder"])
    print("Plot & Save are finished!")
