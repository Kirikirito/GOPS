from gops.env.env_ocp.resources import lq_configs
from gops.env.env_ocp.resources.lq_base import LqModel


def env_model_creator(**kwargs):
    """
    make env model `pyth_linearquadratic`
    """
    lqc = kwargs.get("lq_config", None)
    if lqc is None:
        config = lq_configs.config_s3a1
    elif isinstance(lqc, str):
        assert hasattr(lq_configs, "config_" + lqc)
        config = getattr(lq_configs, "config_" + lqc)
    elif isinstance(lqc, dict):
        config = lqc

    else:
        raise RuntimeError("lq_config invalid")

    return LqModel(config)