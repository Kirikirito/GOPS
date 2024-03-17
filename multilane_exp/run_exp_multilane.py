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

exp_name = 'idsim_multilane_exp_0308_13' 
exp_discription = 'based on exp DSACT_exp0221,mix_map, random ref_v, nominal acc(-2.5m/s^2, when close front car), lane info in sur punish, increase random dacc range(-2.5m/s^2,-0.5m/s^2), use actor to update pi, add small car in the front 10 scene, period:0.5->0.75, per_buffer alpha0.6, LRdecay, full scene evluation, max_step 200, enable slow reward, with opposite direct cars, random takeover acc and steer, fix rear veh punish threshold, random seed in vecenv,change nomimal acc reward design (6->4), increase random dacc time expand ref_v eange to (0-10), fix per buffer decrease dacc action rate, overspead punish, dynamic living reward, sifety margin 1, explore action noise std 0.1, stratified buffer, sceneraio based sapling, obs(8,2,4), buffer size 225000, new brake condition, cancel lat and heading punish when low ref_v braking (reserve ref_v punish), when emergency brake, cancel ref_v, sample ratio 5->4 add boundary obs, mlp_seperated, punish boundary 0'


script_folder = "dsac"
algs = ['dsact']
apprfuncs = ['pi']
envs = ['idsim_multilane_vec']
repeats_num = 1
surfix_filter = 'offserial.py'
run_config = {
    # 'env_id': ['gym_carracingraw'],
    'seed':[12345],
    'buffer_max_size':[225000],
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
    