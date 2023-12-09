#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: run a closed-loop system
#  Update: 2022-12-05, Congsheng Zhang: create file


from gops.sys_simulator.sys_run import PolicyRunner

runner = PolicyRunner(
    log_policy_dir_list=["/home/gaojiaxin/gops_carracing/GOPS/exp/temp/dsact_cnn/gym_carracingraw_12345_main_exp_1112_run0",
                         "/home/gaojiaxin/gops_carracing/GOPS/exp/temp/ppo_cnn/gym_carracingraw_12345_main_exp_1112_run0",
                         "/home/gaojiaxin/gops_carracing/GOPS/results/carracingraw/trpo_cnn/gym_carracingraw_12345_main_exp_1112_run0",
                         "/home/gaojiaxin/gops_carracing/GOPS/exp/temp/dsac_cnn/gym_carracingraw_12345_main_exp_1112_run0",
                         "/home/gaojiaxin/gops_carracing/GOPS/results/carracingraw/sac_cnn/gym_carracingraw_12345_main_exp_1112_run0"],
    trained_policy_iteration_list=["330000_opt",
                                   "495_opt",
                                   "495_opt",
                                   "160000_opt",
                                   "450000_opt"],
    is_init_info=False,
    init_info={"init_state": [-1, 0.05, 0.05, 0, 0.1, 0.1]},
    save_render=True,
    legend_list=["DSACT", "PPO","TRPO","DSAC","SAC"],
    use_opt=False,  # Use optimal solution for comparison
    dt=0.01, # time interval between steps
)

runner.run()
