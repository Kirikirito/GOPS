from gops.create_pkg.create_env import create_env
from gops.env.tools.env_dynamic_checker import check_dynamic


check_dynamic(env_info={'env_id':'pyth_lq', 'lq_config':'s2a1'},
              log_policy_dir='./results/INFADP/s2a1',
              policy_iteration='350000')