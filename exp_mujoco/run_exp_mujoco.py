import argparse
import os
import copy
import json
import logging
from exp_runner import BaseExpRunner



base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Experiment parameters
script_path = os.path.join(base_path, 'example_train')
save_folder = os.path.join(base_path, 'results/exp_dsact')

exp_name = 'exp_dsact_10_smonet_vec_loss_weight_17_16_smonet4_5' 
exp_discription = 'new q loss form5 clip 0,10, fix clip bug, not seq_input for q, method 5, smonet4, new weight bias form (constant=10) rel noise0.1 decay 0.01, new buffer random level, freeze policy, fix noise in obs for alg, attn_head num = 8, attn_layer=2, high learning rate, no jacobi reg, weight 1e-4, no any freeze'


script_folder = "dsact"
algs = ['dsact']
apprfuncs = ['smonet4']
envs = ['mujoco']
repeats_num = 1
surfix_filter = 'vecoffserial.py'
run_config = {
    'env_id': ['gym_walker2d','gym_ant', 'gym_humanoid','gym_halfcheetah','gym_reacher'],
    'seed':[12345],
    # 'seq_len':[4],
    # 'loss_weight':[0.0,1e-4],

    # 'eval_interval':[2000],
    # 'sample_batch_size':[10],
    # 'sample_interval':[1],
    # 'reward_scale':[1],
    # 'vector_env_num':[10],
    # 'max_iteration':[10000],
    # 'log_save_interval':[500],
    # 'buffer_warm_size':[1000],
}

project_root = None
save_meta_data = True
save_zip = True
max_subprocess = 5
max_waiting_time = 96 * 3600  # seconds
log_level = 'DEBUG'


paser = argparse.ArgumentParser()
paser.add_argument('--max_subprocess', type=int, default=max_subprocess)
paser.add_argument('--max_waiting_time', type=int, default=max_waiting_time)
paser.add_argument('--script_folder', type=str, default=script_folder)
paser.add_argument('--algs', type=list, default=algs)
paser.add_argument('--apprfuncs', type=list, default=apprfuncs)
paser.add_argument('--envs', type=list, default=envs)
paser.add_argument('--repeats_num', type=int, default=repeats_num)
paser.add_argument('--save_folder', type=str, default=save_folder)
paser.add_argument('--script_path', type=str, default=script_path)
paser.add_argument('--run_config', type=dict, default=run_config)
paser.add_argument('--exp_name', type=str, default=exp_name)
paser.add_argument('--exp_discription', type=str, default=exp_discription)
paser.add_argument('--save_meta_data', type=bool, default=save_meta_data)
paser.add_argument('--save_zip', type=bool, default=save_zip)
paser.add_argument('--project_root', type=str, default=project_root)
paser.add_argument('--surfix_filter', type=str, default=surfix_filter)
paser.add_argument('--log_level', type=str, default=log_level)
args = paser.parse_args()



if __name__ == '__main__':
    args = vars(args)
    exp_runner = BaseExpRunner(**args)
    with open(os.path.join(exp_runner.save_folder, 'exp_config.json'), 'w') as f:
        json.dump(args, f, indent=4)
    exp_runner.run()
    