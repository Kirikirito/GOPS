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
import os
def get_policy_dir_iter_list():
    envs = ["humanoid"]
    algs = ["DSAC"]
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'results')
    policy_dir_list = []
    polcicy_iter_list = []
    legend_list = []
    for env in envs:
        for alg in algs:
            legend_list.append(env + "_" + alg)
            tar_dir = os.path.join(base_dir,env,alg)
            if os.path.exists(tar_dir):
                sub_dir = list(os.listdir(tar_dir))[0]
                policy_dir_list.append(os.path.join(tar_dir,sub_dir))
                appr_dir = os.path.join(tar_dir,sub_dir,"apprfunc")
                for file in os.listdir(appr_dir):
                    if file.endswith("_opt.pkl"):
                        polcicy_iter_list.append(file.split("_")[1] + "_opt")

    return policy_dir_list,polcicy_iter_list, legend_list


policy_dir_list, polcicy_iter_list, legend_list = get_policy_dir_iter_list()
for i in range(len(policy_dir_list)):
    runner = PolicyRunner(log_policy_dir_list=[policy_dir_list[i]],
                            trained_policy_iteration_list=[polcicy_iter_list[i]],
                            is_init_info=False,
                            legend_list=[legend_list[i]],
                            save_render=True,
                            use_opt=False,)
    runner.run()


