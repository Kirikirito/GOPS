from argparse import ArgumentParser
from utils import parallel_eval
from easydict import EasyDict as edict

test_conf= edict()
test_conf.render_only = False
test_conf.generate_animation = True
test_conf.sur_bias_range = [1.0, 1.5]
test_conf.sur_bias_prob = 0.5
test_conf.ini_network_root = r"/root/gops/results/idsim/idsim_multilane_exp_0813_2/idsim_multilane_vec/dsact_pi/12345_2000000_run0"
test_conf.nn_index = '350000'
test_conf.dpi = 60
test_conf.test_name = 'withsur'
test_configs_file = '/root/gops/idsim_run/test_configs/multilane_withsur.json'


if __name__ == "__main__":
    parallel_eval(test_conf, test_configs_file)
