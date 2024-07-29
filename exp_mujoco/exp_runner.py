import os
import subprocess
import itertools
import time
import json
import logging
import datetime
import warnings
import copy
import inspect

import numpy as np


class BaseExpRunner:
    def __init__(self, script_path, script_folder, algs, apprfuncs, envs, repeats_num, run_config, surfix_filter = 'serial.py',
                 max_subprocess = 1, max_waiting_time = 48 * 3600,
                  log_level = 'DEBUG', save_folder= './exp_results', exp_name=None,exp_discription='', project_root=None,save_meta_data= True, save_zip=True, **kwargs):
        
        self.script_path = script_path
        if exp_name is None:
            self.exp_name = datetime.datetime.now().strftime(r"%y%m%d-%H%M%S")
        else:
            self.exp_name = exp_name
        self.save_folder = os.path.join(save_folder, self.exp_name)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.script_folder = script_folder
        self.algs = algs
        self.apprfuncs = apprfuncs
        self.envs = envs
        self.repeats_num = repeats_num
        self.configs = self.prodecut_config(run_config)  
        self.surfix_filter = surfix_filter  
        self.logger = self.get_logger(log_level, self.save_folder)
        self.exp_discription = exp_discription
        self.max_subprocess = max_subprocess
        self.max_waiting_time = max_waiting_time
        self.project_root = project_root
        self.save_meta_data = save_meta_data
        self.save_zip = save_zip



    def prodecut_config(self, configs: dict):
        keys, values = zip(*configs.items())
        result = []
        for comb in itertools.product(*values):
            result.append(dict(zip(keys, comb)))
        return result
    def parse_config(self, file_path: str, config: dict):
        command_string = 'python ' + file_path
        for key, value in config.items():
            command_string += f' --{key} {value}'
        return command_string

    def get_tar_file(self, tar_dir, alg, appr, env):
        tar_file = None
        for file in os.listdir(tar_dir):
            if file.startswith('_'.join([alg, appr, env])) and file.endswith(self.surfix_filter):
                tar_file = file
                break
        return tar_file
    
    def get_logger(self, log_level, save_floder):
        log_file = os.path.join(save_floder, 'exp_logger.txt')
        selflogger = logging.getLogger('Experiment Logger')
        logging.basicConfig(filename=log_file,
                            filemode='a',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                            level=log_level)
        return selflogger

    def run_scripts(self, file_paths, configs: dict, repeats):
        logger = self.logger 
        p_pool = []
        for idx in range(len(file_paths)):
            file_path = file_paths[idx]
            config = configs[idx]
            local_config = config.copy()
            repeat = repeats[idx]
            local_config['save_folder'] = config['save_folder'] + '_run' + str(repeat)
            command_string = self.parse_config(file_path, local_config)
            p = subprocess.Popen(command_string, close_fds=True, shell=True)
            p_pool.append(p)

        self.monitor_processes(p_pool)
        return
    
    def monitor_processes(self, p_pool):
        start_time = time.time()
        logger = self.logger    
        while True:
            for p in p_pool:
                if p.poll() is not None:
                    if p.poll() == 0:
                        logger.debug('\nProcess of command:\n {} \n exits successfully!'.format(p.args))
                    else:
                        logger.error(
                            '\nProcess of command:\n {} \n exits with error code {}!'.format(p.args, p.poll()))
                    p_pool.remove(p)

            if not p_pool:
                logger.debug('All processes of current run exit!')
                break

            current_time = time.time()
            if current_time > start_time + self.max_waiting_time:
                logger.error('\nProcesses of \n{} \n has exceed max waiting time!'.format([p.args for p in p_pool]))
                break

            time.sleep(10)
        


    def run(self):
        script_path = self.script_path
        script_floder = self.script_folder
        algs = self.algs
        apprfuncs = self.apprfuncs
        envs = self.envs
        repeats_num = self.repeats_num
        repeat_groups = list(range(repeats_num))
        config_groups = self.configs
        logger = self.logger
        # used to store the full file path of the scripts
        full_file_paths = []
        configs = []
        repeats = []
        num_subprocess = 0
        ruturn_val = git_backup(self.save_folder, self.project_root, self.exp_discription, self.save_meta_data, self.save_zip)
        if not ruturn_val:
            return
        else:
            logger.info("Git backup successfully!")
            print("Git backup successfully!")

        for alg, appr, env, config_group, repeat in itertools.product(algs, apprfuncs, envs, config_groups, repeat_groups):
            if script_floder is None:
                real_script_floder = alg
            else:
                real_script_floder = script_floder
            tar_dir = os.path.join(script_path, real_script_floder)
            tar_file = self.get_tar_file(tar_dir, alg, appr, env)

            if tar_file:
                logger.info("run {} of script: ".format(str(repeat)) + tar_file)
                print("run {} of script: ".format(str(repeat)) + tar_file)
                full_file_path = os.path.join(tar_dir, tar_file)
                config_dir = '_'.join(str(val).replace(' ', '_') for val in config_group.values())
                config = {'save_folder': os.path.join(self.save_folder, env, '_'.join([alg, appr]), config_dir)}    
                config.update(config_group)
                logger.info("with config: {}".format(config))
                print("with config: {}".format(config))

                if num_subprocess < self.max_subprocess:
                    full_file_paths.append(full_file_path)
                    configs.append(config)
                    repeats.append(repeat)
                    num_subprocess += 1

                if num_subprocess == self.max_subprocess:
                    self.run_scripts(full_file_paths, configs, repeats)
                    num_subprocess = 0
                    full_file_paths = []
                    configs = []
                    repeats = []

            else:
                logger.info("missing script for {}_{}_{} and ends with {} in dir {}! ".format(alg, appr, env, self.surfix_filter, tar_dir))
                print("missing script for {}_{}_{} and ends with {} in dir {}! ".format(alg, appr, env, self.surfix_filter, tar_dir))
                

        if num_subprocess > 0:
            self.run_scripts(full_file_paths, configs, repeats)

        logger.info("Finish running scripts!")



def run_scripts(file_paths, configs: dict, repeats, logger):
    p_pool = []
    for idx in range(len(file_paths)):
        file_path = file_paths[idx]
        config = configs[idx]
        local_config = config.copy()
        repeat = repeats[idx]
        local_config['save_folder'] = config['save_folder'] + '_run' + str(repeat)
        command_string = parse_config(file_path, local_config)
        p = subprocess.Popen(command_string, close_fds=True, shell=True)
        p_pool.append(p)

    monitor_processes(p_pool, logger, repeats)
    return


def monitor_processes(p_pool, logger):
    start_time = time.time()
    while True:
        for p in p_pool:
            if p.poll() is not None:
                if p.poll() == 0:
                    logger.debug('\nProcess of command:\n {} \n exits successfully!'.format(p.args))
                    print('\nProcess of command:\n {} \n exits successfully!'.format(p.args))
                else:
                    logger.error(
                        '\nProcess of command:\n {} \n exits with error code {}!'.format(p.args, p.poll()))
                    warnings.warn(
                        '\nProcess of command:\n {} \n exits with error code {}!'.format(p.args, p.poll()))
                p_pool.remove(p)

        if not p_pool:
            logger.debug('All processes of current run exit!')
            print('All processes of current run exit!')
            break

        current_time = time.time()
        if current_time > start_time + max_waiting_time:
            logger.error('\nProcesses of \n{} \n has exceed max waiting time!'.format([p.args for p in p_pool]))
            warnings.warn('\nProcesses of \n{} \n has exceed max waiting time!'.format([p.args for p in p_pool]))
            break

        time.sleep(10)


def parse_config(file_path: str, config: dict):
    command_string = 'python ' + file_path
    for key, value in config.items():
        command_string += f' --{key} {value}'
    print(command_string)
    return command_string


def product_config(configs: dict):
    keys, values = zip(*configs.items())
    result = []
    for comb in itertools.product(*values):
        result.append(dict(zip(keys, comb)))
    return result



def get_repo_changes(repo, save_folder):
    changes = []
    patch_file = os.path.join(save_folder, 'changes.patch')
    with open(patch_file, 'w') as patch:
        repo.git.add("-A", ".")
        diff_summary = repo.git.diff("--stat", "--cached")
        changes.append(("Main Repository", diff_summary))
        print(f"\nChanges in Main Repository:\n{diff_summary}")
        diff_details = repo.git.diff("--minimal", "--patience", "--cached")
        patch.write(f"\nMain Repository Details:\n{diff_details}\n")
        for submodule in repo.submodules:
            submodule_path = submodule.path
            submodule_repo = submodule.module()
            submodule_repo.git.add("-A", ".")
            submodule_diff_summary = submodule_repo.git.diff("--stat", "--cached")
            changes.append((f"Submodule: {submodule_path}", submodule_diff_summary))
            print(f"\nChanges in Submodule: {submodule_path}:\n{submodule_diff_summary}")
            diff_details = submodule_repo.git.diff("--minimal", "--patience", "--cached")
            patch.write(f"\nSubmodule: {submodule_path} Details:\n{diff_details}\n")
    os.chmod(patch_file, 0o444) # make it read only
    return changes

def git_backup(save_folder, project_root, exp_discription,save_meta_data= True, save_zip=True):
    try:
        from git import Repo
        from git.exc import InvalidGitRepositoryError
    except ImportError as e:
        print(f"Can't import `git`: {e}")
        return

    try:
        if project_root is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            print(f"Use default project root: {project_root}")
        repo = Repo(project_root)
        if save_meta_data:
            changes = get_repo_changes(repo, save_folder)

            exp_summary = os.path.join(save_folder, 'exp_summary.txt')
            meta_data = {}
            repo_dict = {}
            repo_dict['path'] = project_root

            with open(exp_summary, 'w') as f:
                f.write(f"Experiment Discription: {exp_discription}\n")
                current_branch = repo.active_branch
                repo_dict['branch'] = current_branch.name
                print(f"Current branch in main repository: {current_branch}")
                f.write(f"Current branch in main repository: {current_branch}\n")

                current_commit_id = repo.head.commit.hexsha
                repo_dict['commit_id'] = current_commit_id
                print(f"Current commit ID in main repository: {current_commit_id}")
                f.write(f"Current commit ID in main repository: {current_commit_id}\n")
                meta_data['main_repo'] = repo_dict

                submodules = repo.submodules

                for submodule in submodules:
                    repo_dict = {}
                    submodule_path = submodule.path
                    submodule_repo = submodule.module()
                    repo_dict['path'] = os.path.join(project_root, submodule_path)


                    submodule_branch = submodule_repo.active_branch
                    repo_dict['branch'] = submodule_branch.name
                    print(f"Branch in submodule '{submodule_path}': {submodule_branch}")
                    f.write(f"Branch in submodule '{submodule_path}': {submodule_branch}\n")

                    submodule_commit_id = submodule_repo.head.commit.hexsha
                    repo_dict['commit_id'] = submodule_commit_id
                    print(f"Commit ID in submodule '{submodule_path}': {submodule_commit_id}")
                    f.write(f"Commit ID in submodule '{submodule_path}': {submodule_commit_id}\n")

                    meta_data[submodule_path] = repo_dict
            


                for change_detail in changes:
                    f.write(f"\n{change_detail[0]}:\n")
                    f.write(change_detail[1])
            os.chmod(exp_summary, 0o444) # make it read only
            
            meta_data_file = os.path.join(save_folder, 'meta_data.json')
            with open(meta_data_file, 'w') as f:
                json.dump(meta_data, f, indent=4, default=change_type)
            os.chmod(meta_data_file, 0o444) # make it read only
        
        val = input("Press Enter to continue, or press 'esc' to exit: ")
        if val == '\x1b':
            print("Exit")
            return False
        else:
            print("Continue")
        
        if save_zip:
            zip_path = os.path.join(save_folder, 'git_backup.zip')
            from zipfile import ZipFile
            pkg = ZipFile(zip_path, 'w')

            for file_name in repo.git.ls_files().split():
                # exclude  all files in results folder
                if file_name.startswith('results'):
                    continue
                # exclude all slx and pyd files
                if file_name.endswith('.slx') or file_name.endswith('.pyd'):
                    continue

                pkg.write(os.path.join(project_root, file_name), arcname=file_name)

            for submodule in repo.submodules:
                submodule_path = submodule.path
                submodule_repo = submodule.module()
                for file_name in submodule_repo.git.ls_files().split():
                    # exclude files that not exists
                    if not os.path.exists(os.path.join(project_root,submodule_path, file_name)):
                        warnings.warn(f"File {file_name} in submodule {submodule_path} does not exist! Skip it.")
                        continue
                    pkg.write(os.path.join(project_root,submodule_path, file_name), arcname=os.path.join(submodule_path, file_name))

            pkg.close()
        
        return True


    except InvalidGitRepositoryError as e:
        import traceback
        traceback.print_exc()
        print(f"Can't use git to backup files: {e}")
    except FileNotFoundError as e:
        import traceback
        traceback.print_exc()
        print(f"Can't find file {e}. Did you delete a file and forget to `git add .`")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Unknown error: {e}")

def change_type(obj):
    if isinstance(
        obj,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)
    elif isinstance(obj, type):
        return str(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = change_type(v)
        return obj
    elif isinstance(obj, list):
        for i, o in enumerate(obj):
            obj[i] = change_type(o)
        return obj
    elif isinstance(obj, tuple):
        return obj


base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
script_path = os.path.join(base_path, 'example_train')
save_path = os.path.join(base_path, 'results')

# Experiment parameters
max_subprocess = 1
max_waiting_time = 48 * 3600  # seconds
script_floder = "dsac"
algs = ['dsac']
apprfuncs = ['mlp']
# envs = ['idsim_multilane','idsim_multilane_vec']
envs = ['cartpoleconti']
repeats_num = 1

run_config = {
    'algorithm': ['DSACT'],
    'max_iteration':[2000],
}


if __name__ == "__main__":
    exp_runner = BaseExpRunner(script_path, script_floder, algs, apprfuncs, envs, repeats_num, run_config, surfix_filter = 'serial.py',
                 max_subprocess = 1, max_waiting_time = 48 * 3600,
                  log_level = logging.INFO, save_folder= save_path, exp_name=None,exp_discription='')
    exp_runner.run()
