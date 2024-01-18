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

exp_name = 'idsim_multilane_exp_0118_1' 
exp_discription = 'based on exp DSACT_exp0115_3, new map(random sur veh size and speed)), tar-pi, low alpha learning rate, random acc (small prob) and take over bias(small phi) '

script_folder = "dsac"
algs = ['dsact']
apprfuncs = ['pi']
envs = ['idsim_multilane_vec']
repeats_num = 1
surfix_filter = 'offserial.py'
run_config = {
    # 'env_id': ['gym_carracingraw'],
    'seed':[12345],
    'buffer_max_size':[500000],
    # 'eval_interval':[2000],
    # 'sample_batch_size':[10],
    # 'sample_interval':[1],
    # 'reward_scale':[1],
    # 'vector_env_num':[10],
    # 'max_iteration':[200000],
}

project_root = None
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
    os.chmod(os.path.join(exp_runner.save_folder, 'exp_config.json'), 0o444) # set the file to read only
    exp_runner.run()
    