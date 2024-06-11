import argparse
import os
import copy
import json
import logging
from exp_runner import BaseExpRunner



base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Experiment parameters
script_path = os.path.join(base_path, 'example_train')
save_folder = os.path.join(base_path, 'results/idsim')

exp_name = 'idsim_multilane_exp_0529_only_multilane_5' 
exp_discription = 'enable multihead attntion had =8, low tau max iter=2_000_000, new veh param add boundary low safe margin, buffermax_size, new env reward and config (positive reward ending condition category critic and actor update pi) clip lower bound -0.05 add max num of bike and pedestrian add boundaryh obs(abs value), new map(no sur_obs) new pinet(add value mask) fix bug in sur obs filter, add direction selector, low time constant for front vehicle fix boundary obs in junction,change live reward form, nomimal acc 2.5->1,5 only multilane, change corresponding configs, low tracking reward, clip boundary -5, no attn, no boundary, small random ref_v dynamic ref_v_lane, time_dist_coff:1.5->1, enable attn'


script_folder = "dsac"
algs = ['dsact']
apprfuncs = ['pi']
envs = ['idsim_multilane_vec']
repeats_num = 1
surfix_filter = 'offserial.py'
run_config = {
    # 'env_id': ['gym_carracingraw'],
    'seed':[12345],
    'buffer_max_size':[1000000],
    # 'eval_interval':[2000],
    # 'sample_batch_size':[10],
    # 'sample_interval':[1],
    # 'reward_scale':[1],
    # 'vector_env_num':[10],
    # 'max_iteration':[200000],
}

project_root = None
save_meta_data = True
save_zip = True
max_subprocess = 1
max_waiting_time = 48 * 3600  # seconds
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
    