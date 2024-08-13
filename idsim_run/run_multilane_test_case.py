from argparse import ArgumentParser
from utils import parallel_eval
from easydict import EasyDict as edict

test_conf= edict()
test_conf.render_only = False
test_conf.generate_animation = True
test_conf.ini_network_root = r"/root/gops/results/idsim/idsim_multilane_exp_0729_4/idsim_multilane_vec/dsact_pi/12345_2000000_run0"
test_conf.nn_index = '750000'
test_conf.dpi = 60
test_conf.test_name = 'new_test'
test_configs_file = '/root/gops/idsim_run/test_configs/multilane_tracking.json'


if __name__ == "__main__":
    parallel_eval(test_conf, test_configs_file)
