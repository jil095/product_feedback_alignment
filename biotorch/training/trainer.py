"""
Licensed under the Apache License, Version 2.0
Modified by: PFA's authors
Originally created by: jsalbert (https://github.com/jsalbert/biotorch)
"""
import os
import time
import torch


from torch.utils.tensorboard import SummaryWriter
from biotorch.training.functions import train, test
from biotorch.training.metrics import compute_angles_module, compute_weight_ratio_module, compute_weight_norm_module


class Trainer:
    def __init__(self,
                 model,
                 mode,
                 loss_function,
                 optimizer,
                 lr_scheduler,
                 wd_scheduler,
                 train_dataloader,
                 val_dataloader,
                 device,
                 epochs,
                 output_dir,
                 metrics_config,
                 multi_gpu=False
                 ):

        self.model = model
        self.mode = mode
        self.output_dir = output_dir
        self.logs_dir = os.path.join(output_dir, 'logs')
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.wd_scheduler = wd_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.epochs = epochs
        self.multi_gpu = multi_gpu
        self.display_iterations = metrics_config['display_iterations']
        self.record_layer_alignment = metrics_config['layer_alignment']
        self.record_weight_ratio = metrics_config['weight_ratio']
        self.record_weight_norm = True
        self.top_k = metrics_config['top_k']
        self.writer = SummaryWriter(self.logs_dir)
        self.layer_alignment_modes = ['fa', 'usf', 'frsf', 'brsf','tfa']

    def write_layer_alignment(self, epoch):
        if self.record_layer_alignment:
            if self.mode in self.layer_alignment_modes:
                # Fails for multi-gpu and usf, brsf as weight backwards is being written in replicas
                try:
                    layers_alignment = compute_angles_module(self.model)
                    self.writer.add_scalars('layer_alignment/train', layers_alignment, epoch)
                except BaseException as e:
                    print('Layer alignment could not be computed for {}'.format(self.mode))
                    raise e
            else:
                print('Layer alignment is not implemented for  {}'.format(self.mode))

    def write_weight_ratio(self, epoch):
        if self.record_weight_ratio:
            try:
                weight_ratio = compute_weight_ratio_module(self.model, self.mode)
                self.writer.add_scalars('weight_radio/train', weight_ratio, epoch)
            except BaseException as e:
                print('Weight ratio could not be computed for {}'.format(self.mode))
                raise e

    def write_weight_norm(self, epoch):
        if self.record_weight_norm:
            try:
                weight_norm = compute_weight_norm_module(self.model, self.mode)
                self.writer.add_scalars('weight_norm/train', weight_norm, epoch)
            except BaseException as e:
                print('Weight norm could not be computed for {}'.format(self.mode))
                raise e

    def run(self):
        self.best_acc = 0.0

        for epoch in range(self.epochs):
            self.write_layer_alignment(epoch)
            self.write_weight_ratio(epoch)
            self.write_weight_norm(epoch)

            t = time.time()
            acc, loss = train(
                model=self.model,
                mode=self.mode,
                loss_function=self.loss_function,
                optimizer=self.optimizer,
                train_dataloader=self.train_dataloader,
                device=self.device,
                multi_gpu=self.multi_gpu,
                epoch=epoch,
                top_k=self.top_k,
                display_iterations=self.display_iterations
            )

            self.writer.add_scalar('accuracy/train', acc, epoch)
            self.writer.add_scalar('loss/train', loss, epoch)

            acc, loss = test(
                model=self.model,
                loss_function=self.loss_function,
                test_dataloader=self.val_dataloader,
                device=self.device,
                top_k=self.top_k
            )
            self.writer.add_scalar('accuracy/test', acc, epoch)
            self.writer.add_scalar('loss/test', loss, epoch)

            # Remember best acc@1 and save checkpoint
            if acc > self.best_acc:
                self.best_acc = max(acc, self.best_acc)
                print('New best accuracy reached: {} \nSaving best accuracy model...'.format(self.best_acc))
                if self.multi_gpu:
                    torch.save(self.model.module, os.path.join(self.output_dir, 'model_best_acc.pth'))
                else:
                    torch.save(self.model, os.path.join(self.output_dir, 'model_best_acc.pth'))
            torch.save(self.model, os.path.join(self.output_dir, 'latest_model.pth'))

            total_time = time.time() - t

            # Update scheduler after training epoch
            self.lr_scheduler.step()
            self.wd_scheduler.step()
            self.writer.add_scalar('time/train', total_time, epoch)

        with open(os.path.join(self.output_dir, 'best_acc.txt'), 'w') as f:
            f.write(str(self.best_acc))

        self.write_layer_alignment(epoch)
        self.write_weight_ratio(epoch)
        self.write_weight_norm(epoch)

