from argparse import ArgumentParser
from utils import parallel_eval, generate_noise_data, multi_parallel_eval, combine_conf
from easydict import EasyDict as edict


test_conf= edict()
test_conf.render_only = False
test_conf.generate_animation = False
test_conf.tracking_only = False
test_conf.ini_network_root = r"/root/gops/results/idsim/idsim_multilane_exp_0729_4/idsim_cross_vec/dsact_pi/12345_2000000_run0"
test_conf.nn_index = '750000'
test_conf.dpi = 60
test_conf.noise_type = 'uniform'
test_conf.test_name = 'noise_test_sur'
test_configs_file = '/root/gops/idsim_run/test_configs/cross_noise_test.json'


conf_list = edict()
conf_list.noise_data = generate_noise_data(test_conf.noise_type, num = 10, scale = 'log', low = 5e-3, high = 5e-1)
conf_list.ini_network_root = [  r"/root/gops/results/idsim/idsim_compare/idsim_cross_vec/dsact_mlp/12345_1500000_run0",
                                r"/root/gops/results/idsim/idsim_compare/idsim_cross_vec/dsact_pi/12345_2000000_run0",
                                r"/root/gops/results/idsim/idsim_compare/idsim_cross_vec/dsact_pismonet/12345_2000000_run0",
                                r"/root/gops/results/idsim/idsim_compare/idsim_cross_vec/dsac_mlp/12345_1500000_run0",
                                r"/root/gops/results/idsim/idsim_compare/idsim_cross_vec/sac_mlp/12345_1500000_run0",
                                    ]
conf_list.nn_index = ['250000', 
                      '1950000',
                        '1950000',
                        '250000',
                        '250000',
                      ]
legends = ['DSACT-MLP', 'DSACT-PI', 'DSACT-PISMONET', 'DSAC-MLP', 'SAC-MLP']
test_conf_list = combine_conf(test_conf, conf_list)
save_floder = '/root/gops/results/idsim/compare'


if __name__ == "__main__":
    multi_parallel_eval(test_conf_list, legends, test_configs_file, save_floder)
