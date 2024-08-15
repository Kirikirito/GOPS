from argparse import ArgumentParser
from utils import parallel_eval, generate_noise_data, multi_parallel_eval, combine_conf
from easydict import EasyDict as edict


test_conf= edict()
test_conf.render_only = False
test_conf.generate_animation = False
test_conf.ini_network_root = r"/root/gops/results/idsim/idsim_multilane_exp_0729_4/idsim_multilane_vec/dsact_pi/12345_2000000_run0"
test_conf.nn_index = '750000'
test_conf.dpi = 60
test_conf.noise_type = 'uniform'
test_conf.test_name = 'noise_test'
test_configs_file = '/root/gops/idsim_run/test_configs/multilane_tracking.json'


conf_list = edict()
conf_list.noise_data = generate_noise_data(test_conf.noise_type, num = 10, scale = 'log', low = 5e-3, high = 5e-1)
conf_list.ini_network_root = [  r"/root/gops/results/idsim/idsim_multilane_exp_0729_4/idsim_multilane_vec/dsact_pi/12345_2000000_run0",
                              r"/root/gops/results/idsim/idsim_multilane_exp_0729_4_fixbug_last_seq_2_no_punish/idsim_multilane_vec/dsact_pismonet/12345_2000000_run0",
                                    r"/root/gops/results/idsim/idsim_multilane_exp_0729_4_fixbug_last_seq_punish_2/idsim_multilane_vec/dsact_pismonet/12345_2000000_run0",
                                    ]
conf_list.nn_index = ['950000', 
                      '950000',
                        '950000',
                      ]

test_conf_list = combine_conf(test_conf, conf_list)

if __name__ == "__main__":
    multi_parallel_eval(test_conf_list, test_configs_file)
