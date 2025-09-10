"""
Licensed under the Apache License, Version 2.0
Created by: PFA's authors
"""
import os
import matplotlib.pyplot as plt
from tbparse import SummaryReader
from pathlib import Path
from utils import *
import pandas as pd

from matplotlib.cm import get_cmap

def get_ann_layer_path(dataset):
    parent_path = Path(r'..\trained_models')
    if dataset == 'mnist':
        ann = 'ffn2'
        layer_list = ['fc1_0', 'fc2_0', 'fc_0']
        parent_path = parent_path / 'mnist' / ann
        xlim, xticks = 40, [0, 20, 40]
        seed_suffixs = ['_seed0', '_seed1', '_seed2', '_seed3', '_seed4']
    elif dataset == 'cifar10':
        ann = 'resnet20'
        layer_list = ['conv1_0']+ sum([[f'conv1_{s+1}', f'conv2_{s}'] for s in range(0, 9)],[]) + ['fc_0']
        # conv1_0: (3 in channels) 16 out channels
        # resnet20_layer1: 3 blocks of [2 sublayers of [16 out channels]]
        # resnet20_layer2: 3 blocks of [2 sublayers of [32 out channels]]
        # resnet20_layer3: 3 blocks of [2 sublayers of [64 out channels]]
        # fc_0: 64 in channels, 10 out channels
        parent_path = parent_path / 'cifar10' / ann
        xlim, xticks = 200, [0, 100, 200]
        seed_suffixs = ['_seed0', '_seed1', '_seed2', '_seed3', '_seed4']
    elif dataset == 'imagenet':
        ann = 'resnet18'
        layer_list = [
            'conv1_0',  # 3 in channels, 64 out channels
            'conv1_1', 'conv2_0','conv1_2','conv2_1', # resnet18_conv2_x: 2 blocks of [2 sublayers of [64 out channels]]
            '0_0','conv1_3', 'conv2_2', 'conv1_4','conv2_3', # resnet18_conv3_x: shortcut + 2 blocks of [2 sublayers of [128 out channels]]
            '0_1', 'conv1_5', 'conv2_4',  'conv1_6', 'conv2_5', # resnet18_conv4_x: shortcut + 2 blocks of [2 sublayers of [256 out channels]]
            '0_2', 'conv1_7', 'conv2_6', 'conv1_8', 'conv2_7', # resnet18_conv5_x: shortcut + 2 blocks of [2 sublayers of [512 out channels]]
            'fc_0', # 512 in channels, 1000 out channels
        ]
        parent_path = parent_path / 'imagenet' / ann
        xlim, xticks = 75, [0, 25, 50, 75]
        seed_suffixs = ['']
    else:
        raise ValueError('Invalid dataset')
    return ann, layer_list, parent_path, xlim, xticks, seed_suffixs

def path2df(folder_path):
    # search for tf event files in folder
    max_size = 0
    max_size_path = None
    for path in folder_path.rglob('*.tfevents.*'):
        path = str(path)
        size = os.path.getsize(path)
        if size > max_size: # only the largest file corresponds to the latest full run
            max_size = size
            max_size_path = path
    if max_size_path is not None:
        path = max_size_path
    else:
        return None
    print(path)
    reader = SummaryReader(path)
    df = reader.scalars
    return df

def get_alg_perf_metric(parent_path, alg_name, layer_list=[], metrics=[]):
    df = path2df(parent_path / alg_name / 'logs')
    if df is None:
        return None
    alg_perf_metric = {}
    alg_perf_metric['train_acc'] = df[df['tag'] == 'accuracy/train']
    alg_perf_metric['test_acc'] = df[df['tag'] == 'accuracy/test']
    for metric_name in metrics:
        for layer in layer_list:
            metric = f'{metric_name}_train_{layer}'
            alg_perf_metric[metric] = path2df(parent_path / alg_name / 'logs' / metric)
    return alg_perf_metric

color_map = {'backpropagation': 'C0', 'fa': 'C1', 'dfa': 'C2', 'tfawd': 'C3', 'usf': 'C4', 'tfawdo': 'C5'}

if __name__ == '__main__':
    exp_name = 'cifar10'
    # exp_name = 'mnist'
    # exp_name = 'imagenet'
    ann, layer_list, parent_path, xlim, xticks, seed_suffixs = get_ann_layer_path(exp_name)
    dfs = {}

    for alg_name in ['backpropagation', 'fa', 'dfa', 'tfawd','usf','tfawdo']:
        if alg_name not in ['fa', 'tfawd', 'tfawdo', 'usf']:
            metrics = []
        else:
            metrics = ['layer_alignment', 'weight_radio']
        for additional_name in seed_suffixs:
            alg_perf_metric = get_alg_perf_metric(parent_path, alg_name + additional_name, layer_list=layer_list, metrics=metrics)
            if alg_perf_metric is not None:
                dfs[alg_name + additional_name] = alg_perf_metric
        if len(seed_suffixs)>1:
            dfs[alg_name] = { # mean & std over seeds
                'train_acc': pd.concat([dfs[alg_name + f'_seed{i}']['train_acc'] for i in range(5)]).groupby('step', as_index=False).agg(
                    step=('step', 'first'),
                    tag=('tag', 'first'),
                    value=('value', 'mean'),
                    value_std=('value', 'std')
                ),
                'test_acc': pd.concat([dfs[alg_name + f'_seed{i}']['test_acc'] for i in range(5)]).groupby('step', as_index=False).agg(
                    step=('step', 'first'),
                    tag=('tag', 'first'),
                    value=('value', 'mean'),
                    value_std=('value', 'std')
                )

            }

    fig, ax = plot_start(figsize=(2,1.5))
    min_acc = 100
    for alg, alg_label in zip(['backpropagation', 'fa', 'dfa','usf', 'tfawd','tfawdo'], ['BP', 'FA', 'DFA','SF', 'PFA','PFA-o']):
        if alg not in dfs:
            continue
        min_acc = min(min_acc, dfs[alg]['test_acc']['value'].min())
        plt.plot(dfs[alg]['test_acc']['step'], dfs[alg]['test_acc']['value'], label=alg_label, color=color_map[alg], linewidth=0.5)
        if 'value_std' in dfs[alg]['test_acc']:
            plt.fill_between(dfs[alg]['test_acc']['step'],
                             dfs[alg]['test_acc']['value']-dfs[alg]['test_acc']['value_std'],
                             dfs[alg]['test_acc']['value']+dfs[alg]['test_acc']['value_std'],
                             alpha=0.5, color=color_map[alg], edgecolor=None)
    plt.xticks(xticks)
    plt.xlim(0, xlim)
    if min_acc> 80:
        plt.yticks([90, 95, 100])
        plt.ylim(90, 100)
    else:
        plt.yticks([0, 20, 40, 60, 80, 100])
        plt.ylim(0, 100)
    plt.xlabel('Epoch')
    plt.ylabel('Test accuracy')
    plt.legend()
    os.makedirs(Path('./figures') / ann, exist_ok=True)
    plt.savefig(Path('./figures') /ann / 'test_acc.pdf', bbox_inches='tight')
    plt.show()

    for alg in ['fa', 'tfawd', 'usf', 'tfawdo']:
        ylabel_prefix = {'fa':'Weight', 'usf':'Weight','tfawd':'Path', 'tfawdo':'Path'}[alg]
        alg += seed_suffixs[0]
        if alg not in dfs:
            continue
        for metric_name in ['layer_alignment', 'weight_radio']:
            fig, ax = plot_start(figsize=(1,1))
            for layer_idx, layer in enumerate(layer_list):
                cmap = get_cmap('viridis')
                layer_color = cmap(layer_idx / len(layer_list))
                metric = f'{metric_name}_train_{layer}'
                plt.plot(dfs[alg][metric]['step'], dfs[alg][metric]['value'], color=layer_color, alpha=0.5, linewidth=0.5
                         )

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            tick = [0, 0.5, 1]
            plt.colorbar(sm, ticks=tick, ax=ax)

            plt.xlim(-xlim/50, xlim)
            plt.xticks(xticks)
            # plt.yticks([0, 0.5, 1])
            plt.ylim({'layer_alignment': (-5, 100), 'weight_radio': (0, 3)}[metric_name])
            plt.yticks({'layer_alignment': [0, 50, 100], 'weight_radio': [0.5, 1, 2]}[metric_name])
            # plt.ylim(0, 1)
            # plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel({'layer_alignment': f'{ylabel_prefix} alignment ($^\circ$)', 'weight_radio': f'{ylabel_prefix} norm ratio'}[metric_name])
            os.makedirs(Path('./figures'), exist_ok=True)
            plt.savefig(Path('./figures') /ann/ f'{alg}_{metric_name}.pdf', bbox_inches='tight')
            plt.show()