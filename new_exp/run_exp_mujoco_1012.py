import os
os.environ["OMP_NUM_THREADS"] = "4"
import re
import subprocess
import itertools
import datetime
import time
import copy
import logging

def get_logger(log_level,log_file):
    selflogger = logging.getLogger('Experiment Logger')
    logging.basicConfig(filename=log_file,
                        filemode= 'a',
                        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                        level=log_level)
    return selflogger

def split_list(lst, n):
    """
    Split a list into sublists with length n
    """
    return [lst[i:i + n] for i in range(0, len(lst), n)]



def get_tar_file(tar_dir,alg,appr,env):
    tar_file = None
    for file in os.listdir(tar_dir):
        if file.startswith('_'.join([alg,appr,env])) and file.endswith('serial.py'):
            tar_file = file
            break
    return tar_file

def run_scripts(file_paths,configs: dict, repeats, logger):
    p_pool = []
    for idx in range(len(file_paths)):
        file_path = file_paths[idx]
        config = configs[idx]
        local_config = config.copy()
        repeat = repeats[idx]
        local_config['save_folder'] = config['save_folder'] + '_run' + str(repeat)
        command_string = parse_config(file_path, local_config)
        p = subprocess.Popen(command_string, close_fds=True,shell=True)
        p_pool.append(p)

    monitor_processes(p_pool,logger)
    return

def monitor_processes(p_pool,logger):

    start_time = time.time()
    while True:
        for p in p_pool:
            if p.poll() is not None:
                if p.poll() == 0:
                    logger.debug('\nProcess of command:\n {} \n exits successfully!'.format(p.args))
                else:
                    logger.error('\nProcess of command:\n {} \n exits with error code {}!'.format(p.args, repeat,p.poll()))
                p_pool.remove(p)

        if not p_pool:
                logger.debug('All processes of current run exit!')
                break

        current_time = time.time()
        if current_time > start_time +max_waiting_time:
            logger.error('\nProcesses of \n{} \n has exceed max waiting time!'.format([p.args for p in p_pool]))
            break

        time.sleep(10)



def parse_config(file_path: str, config : dict):
    command_string = 'python ' + file_path
    for key, value in config.items():
        command_string += ' --'+key+' '+str(value)
    return command_string
    

def product_config(configs: dict):
    keys, values = zip(*configs.items())
    result = []
    for comb in itertools.product(*values):
        result.append(dict(zip(keys, comb)))
    return result


base_path =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
script_path = os.path.join(base_path, 'example_train')
save_path =  os.path.join(base_path, 'results')

# Experiment parameters
max_subprocess = 5
max_waiting_time = 24*3600 # seconds
script_floder = 'dsac2'
algs = ['dsacalgs']
apprfuncs = ['mlp']
envs = ['mujoco']
repeats_num = 1

suffix = 'main_exp_1012'
run_config={
'env_id': ['gym_humanoid','gym_walker2d','gym_halfcheetah','gym_ant','gym_inverteddoublependulum'],
'env_id': ['gym_humanoid'],
'reward_scale': [1.0],
'seed':[12345,22345,32345,42345,52345],
}

log_level = logging.DEBUG
log_file = os.path.join(save_path,suffix+'.txt')

repeat_groups = list(range(repeats_num))
config_groups = product_config(run_config)
logger = get_logger(log_level, log_file)

# used to store the full file path of the scripts
full_file_paths = []
configs = []
repeats = []
num_subprocess = 0


for alg, appr, env,config_group, repeat in itertools.product(algs,apprfuncs,envs,config_groups,repeat_groups):
    if script_floder is None:
        script_floder = alg
    tar_dir = os.path.join(script_path, script_floder)
    tar_file = get_tar_file(tar_dir,alg,appr,env)

    if tar_file:
        logger.info("run {} of script: ".format(str(repeat)) + tar_file)
        full_file_path = os.path.join(tar_dir,tar_file)
        config_dir = '_'.join(str(val).replace(' ','_') for val in config_group.values())
        config = {'save_folder':os.path.join(save_path,env,'_'.join([alg,appr]),'_'.join([config_dir,suffix])) }
        config.update(config_group)
        logger.info("with config: {}".format(config))

        if num_subprocess < max_subprocess: 
            full_file_paths.append(full_file_path)
            configs.append(config)                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
            repeats.append(repeat)
            num_subprocess += 1

        if num_subprocess == max_subprocess:
            run_scripts(full_file_paths,configs, repeats,logger)
            num_subprocess = 0
            full_file_paths = []
            configs = []
            repeats = []

    else:
        logger.info("missing script for {}_{}_{} in dir {}! ".format(alg,appr,env,tar_dir))

if num_subprocess > 0:
    run_scripts(full_file_paths,configs, repeats,logger)

logger.info("Finish running scripts!")
exit(0)

