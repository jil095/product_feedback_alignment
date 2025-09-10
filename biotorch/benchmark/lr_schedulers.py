"""
Licensed under the Apache License, Version 2.0
Originally created by: jsalbert (https://github.com/jsalbert/biotorch)
"""

import torch


def create_lr_scheduler(lr_scheduler_config, optimizer):

    gamma = lr_scheduler_config['gamma']

    if lr_scheduler_config['type'] == 'multistep_lr':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=lr_scheduler_config['milestones'],
            gamma=gamma,
            verbose=True)
    else:
        raise ValueError('Optimizer type {} not supported'.format(lr_scheduler_config['type']))

    return lr_scheduler
