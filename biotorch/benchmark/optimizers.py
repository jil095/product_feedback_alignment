"""
Licensed under the Apache License, Version 2.0
Modified by: PFA's authors
Originally created by: jsalbert (https://github.com/jsalbert/biotorch)
"""

import torch


def create_optimizer(optimizer_config, model):

    lr = optimizer_config['lr']
    weight_decay = optimizer_config['weight_decay']
    momentum = optimizer_config['momentum']

    if optimizer_config['type'] == 'RMSProp':
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=lr,
                                        momentum=momentum,
                                        weight_decay=weight_decay)

    elif optimizer_config['type'] == 'Adam':
        betas = optimizer_config['betas']
        optimizer = torch.optim.AdamW(model.parameters(),
                                     lr=lr,
                                     betas=betas,
                                     weight_decay=weight_decay)

    elif optimizer_config['type'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)
    else:
        raise ValueError('Optimizer type {} not supported'.format(optimizer_config['type']))

    return optimizer
