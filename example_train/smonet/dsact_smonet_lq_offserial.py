#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: example for dsac + humanoidconti + mlp + offserial
#  Update Date: 2021-03-05, Wenxuan Wang: create example
import os
import argparse
import numpy as np

from gops.create_pkg.create_alg import create_alg
from gops.create_pkg.create_buffer import create_buffer
from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_evaluator import create_evaluator
from gops.create_pkg.create_sampler import create_sampler
from gops.create_pkg.create_trainer import create_trainer
from gops.utils.init_args import init_args
from gops.utils.plot_evaluation import plot_all
from gops.utils.tensorboard_setup import start_tensorboard, save_tb_to_csv

os.environ["OMP_NUM_THREADS"] = "4"


if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for users
    parser.add_argument("--env_id", type=str, default="pyth_lq", help="id of environment")
    parser.add_argument("--lq_config", type=str, default="s2a1", help="config of lq")
    parser.add_argument("--algorithm", type=str, default="DSACT", help="RL algorithm")
    parser.add_argument("--enable_cuda", default=False, help="Disable CUDA")
    parser.add_argument("--seed", default=12345, help="Enable CUDA")
    ################################################
    # 1. Parameters for environment
    parser.add_argument("--vector_env_num", type=int, default=4, help="Number of vector envs")
    parser.add_argument("--vector_env_type", type=str, default='async', help="Options: sync/async")
    parser.add_argument("--gym2gymnasium", type=bool, default=True, help="Convert Gym-style env to Gymsnaium-style")

    parser.add_argument("--obs_noise_type", type=str, default= 'normal')
    parser.add_argument("--obs_noise_data", type=float,nargs='+', default= [0, 0.05], help="noise data")
    parser.add_argument("--add_to_info", type=bool, default= True)
    parser.add_argument("--rel_noise_scale", type=bool, default= True)
    parser.add_argument("--augment_act", type=bool,default=False, help="Augment action")
    parser.add_argument("--seq_len", type=int, default=8)
    seq_len = parser.parse_known_args()[0].seq_len

    parser.add_argument("--reward_scale", type=float, default=1, help="reward scale factor")
    parser.add_argument("--is_render", type=bool, default=False, help="Draw environment animation")
    parser.add_argument("--is_adversary", type=bool, default=False, help="Adversary training")

    ################################################
    # 2.1 Parameters of value approximate function
    parser.add_argument("--freeze_q", type=bool, default=True, help="Freeze Q")
    parser.add_argument("--loss_weight", type=float, default=1, help="tau decay factor")
    parser.add_argument("--value_kernel", type=str, default= '1_8_1', help="kernel size")
    parser.add_argument("--policy_kernel", type=str, default= '1_8_1', help="kernel size")
    loss_weight = parser.parse_known_args()[0].loss_weight
    value_kernel = parser.parse_known_args()[0].value_kernel
    value_kernel_size = [int(i) for i in value_kernel.split('_')]
    policy_kernel = parser.parse_known_args()[0].policy_kernel
    policy_kernel_size = [int(i) for i in policy_kernel.split('_')]
    
    parser.add_argument("--tau_layer_num", type=int, default=2, help="Number of tau layers")
    parser.add_argument(
        "--value_func_name",
        type=str,
        default="ActionValueDistri",
        help="Options: StateValue/ActionValue/ActionValueDis/ActionValueDistri",
    )
    parser.add_argument("--value_func_type", type=str, default="SMONET", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS")
    value_func_type = parser.parse_known_args()[0].value_func_type
    parser.add_argument("--value_hidden_sizes", type=list, default=[64,64])
    parser.add_argument(
        "--value_hidden_activation", type=str, default="gelu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--value_output_activation", type=str, default="linear", help="Options: linear/tanh")

    parser.add_argument("--value_kernel_size", type=int,nargs='+', default= value_kernel_size, help="kernel size for each layer")
    parser.add_argument("--value_loss_weight", type=float, default=0.0, help="tau decay factor")

    # 2.2 Parameters of policy approximate function
    parser.add_argument(
        "--policy_func_name",
        type=str,
        default="StochaPolicy",
        help="Options: None/DetermPolicy/FiniteHorizonPolicy/StochaPolicy",
    )
    parser.add_argument(
        "--policy_func_type", type=str, default="SMONET", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS"
    )
    parser.add_argument(
        "--policy_act_distribution",
        type=str,
        default="TanhGaussDistribution",
        help="Options: default/TanhGaussDistribution/GaussDistribution",
    )
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    parser.add_argument("--policy_hidden_sizes", type=list, default=[64,64])
    parser.add_argument(
        "--policy_hidden_activation", type=str, default="gelu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--policy_min_log_std", type=int, default=-20)
    parser.add_argument("--policy_max_log_std", type=int, default=0.5)

    parser.add_argument("--policy_kernel_size", type=int,nargs='+', default= policy_kernel_size, help="kernel size for each layer")
    parser.add_argument("--policy_loss_weight", type=float, default=loss_weight, help="tau decay factor")

    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--value_learning_rate", type=float, default=0.001)
    parser.add_argument("--policy_learning_rate", type=float, default=0.001)
    parser.add_argument("--alpha_learning_rate", type=float, default=0.0003)
    # special parameter
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--auto_alpha", type=bool, default=True)
    parser.add_argument("--alpha", type=bool, default=0.2)
    parser.add_argument("--delay_update", type=int, default=2)

    ################################################
    # 4. Parameters for trainer
    parser.add_argument(
        "--trainer",
        type=str,
        default="off_realserial_trainer",
        help="Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer",
    )
    # Maximum iteration number
    parser.add_argument("--max_iteration", type=int, default=15000)
    parser.add_argument("--freeze_iteration", type=int, default=5000)
    parser.add_argument(
        "--ini_network_dir",
        type=str,
        default=None
    )
    trainer_type = parser.parse_known_args()[0].trainer

    # 4.1. Parameters for off_serial_trainer
    parser.add_argument(
        "--buffer_name", type=str, default="replay_buffer", help="Options:replay_buffer/prioritized_replay_buffer"
    )
    # Size of collected samples before training
    parser.add_argument("--buffer_warm_size", type=int, default=5000)
    # Max size of reply buffer
    parser.add_argument("--buffer_max_size", type=int, default=50000)
    # Batch size of replay samples from buffer
    parser.add_argument("--replay_batch_size", type=int, default=256)
    # Period of sampling
    parser.add_argument("--sample_interval", type=int, default=1)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="off_sampler", help="Options: on_sampler/off_sampler")
    # Batch size of sampler for buffer store
    parser.add_argument("--sample_batch_size", type=int, default=20)
    # Add noise to action for better exploration
    parser.add_argument("--noise_params", type=dict, default=None)

    ################################################
    # 6. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_save", type=str, default=False, help="save evaluation data")
    parser.add_argument("--fixed_eval_seed", type=bool, default=True, help="Fixed evaluation seed")
    parser.add_argument("--fixed_init_state", type=bool, default=True, help="Fixed initial state")
    parser.add_argument("--eval_seed", type=int, default=12345, help="Evaluation seed")
    
    # set train_space & work_space
    parser.add_argument("--initial_distribution", type=str, default="uniform")
    init_mean = np.array([0, 0], dtype=np.float32)
    init_std = np.array([1, 1], dtype=np.float32)
    train_space = np.stack((init_mean - 1 * init_std, init_mean + 1 * init_std))
    work_space = np.stack((init_mean - 0.5 * init_std, init_mean + 0.5 * init_std))
    parser.add_argument("--train_space", type=np.array, default=train_space)
    parser.add_argument("--work_space", type=np.array, default=work_space)

    ################################################
    # 7. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    # Save value/policy every N updates
    parser.add_argument("--apprfunc_save_interval", type=int, default=5000)
    # Save key info every N updates
    parser.add_argument("--log_save_interval", type=int, default=500)

    ################################################
    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**{**args, "vector_env_num": None})
    args = init_args(env, **args)

    #start_tensorboard(args["save_folder"])
    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)
    # Step 2: create sampler in trainer
    sampler = create_sampler(**args)
    # Step 3: create buffer in trainer
    buffer = create_buffer(**args)
    # Step 4: create evaluator in trainer
    evaluator = create_evaluator(**args)
    # Step 5: create trainer
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)

    ################################################
    # Start training ... ...
    trainer.train()
    print("Training is finished!")

    ################################################
    # Plot and save training figures
    #plot_all(args["save_folder"])
    save_tb_to_csv(args["save_folder"])
    print("Plot & Save are finished!")
