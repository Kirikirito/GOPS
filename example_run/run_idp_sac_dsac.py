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
    log_policy_dir_list=[
                        #"/home/wangwenxuan/gops_idp/gops/results/mujoco/sac_mlp/gym_ant_1.0_32345_test_0827_run0",
                        #"/home/wangwenxuan/gops_idp/gops/results/mujoco/sac_mlp/gym_humanoid_1.0_22345_test_0827_run0",
                          "/home/wangwenxuan/gops_idp/gops/results/mujoco/sac_mlp/gym_walker2d_1.0_32345_test_0827_run0"
                         ],
    trained_policy_iteration_list=[
    # "1490000_opt",
    #"1425000_opt",
     "1497500_opt"
        ],
    is_init_info=False,
    init_info={"init_state": [-1, 0.05, 0.05, 0, 0.1, 0.1]},
    save_render=True,
    legend_list=["SAC"],
    use_opt=False,  # Use optimal solution for comparison
    dt=0.01, # time interval between steps
)

runner.run()
