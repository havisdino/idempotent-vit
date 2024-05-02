import argparse
import os
import torch
import yaml


def _dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = _dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def load_config(path='config.yaml'):
    with open(path) as file:
        config = yaml.safe_load(file)
    config = _dict2namespace(config)
    return config


def save(states, epoch, checkpoint_retention):
    file = f'G-{epoch}'
    last_k = epoch - checkpoint_retention
    last_file = f'G-{last_k}'
    if os.path.exists(last_file):
        os.remove(last_file)
    torch.save(states, file)
    