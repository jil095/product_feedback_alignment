"""
Licensed under the Apache License, Version 2.0
Originally created by: jsalbert (https://github.com/jsalbert/biotorch)
"""
import os
import yaml


def read_yaml(yaml_path):
    print(f'Reading YAML file from {yaml_path} at {os.getcwd()}')
    with open(yaml_path, 'r') as f:
        yaml_file = yaml.load(f, Loader=yaml.Loader)
    return yaml_file


def mkdir(path):
    if not os.path.exists(path):
        return os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def path_exists(path):
    if os.path.exists(path):
        return True
    else:
        raise ValueError('Path provided does not exist.')
