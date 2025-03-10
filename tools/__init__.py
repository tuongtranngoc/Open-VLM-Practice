import yaml
import glob
import os
from dotenv import load_dotenv


load_dotenv()
CFG_PATH = 'configs/base.yml'


def load_config(file_path=CFG_PATH):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: config
    """
    ext = os.path.splitext(file_path)[1]
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    with open(file_path, 'rb') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    return config

config = load_config()