import argparse
import os
import torch
from torch import nn
import yaml


def init_weights(m):
    for p in m.parameters():
        nn.init.normal_(p, std=0.04)


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
    dir = 'checkpoints'
    if not os.path.exists(dir):
        os.makedirs(dir)
        
    file = os.path.join(dir, f'G-{epoch}')
    k = epoch - checkpoint_retention
    last_file = os.path.join(dir, f'G-{k}')
    if os.path.exists(last_file):
        os.remove(last_file)
    torch.save(states, file)
    