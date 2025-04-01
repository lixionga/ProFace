import importlib
import os.path as osp


def get_config(config_file):
    # assert config_file.startswith('configs/'), 'config file setting must start with configs/'
    temp_config_name = osp.basename(config_file)
    temp_module_name = osp.splitext(temp_config_name)[0]
    config = importlib.import_module(".configs.base", package='functions.arcface_torch')
    cfg = config.config
    config = importlib.import_module(".configs.%s" % temp_module_name, package='functions.arcface_torch')
    job_cfg = config.config
    cfg.update(job_cfg)
    if cfg.output is None:
        cfg.output = osp.join('work_dirs', temp_module_name)
    return cfg