"""
Licensed under the Apache License, Version 2.0
Created by: PFA's authors
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from biotorch.layers.metrics import compute_matrix_angle
from utils import *
import math
N_out = 100
N_in = 256
expand_ratio = 10
angle_means = []
angle_stds = []
BB_ev_means = []
BB_ev_stds = []
ratios = np.logspace(0,2,11)
density = 1 # conn. density = 1 - sparsity

for expand_ratio in ratios:
    N_mid = int(N_out * expand_ratio)
    weight = nn.Parameter(torch.Tensor(N_out, N_in), requires_grad=False)
    weight_backward_B = nn.Parameter(torch.Tensor(N_out, N_mid), requires_grad=False)
    angles = []
    BB_ev_list = []
    for _ in range(100):
        nn.init.xavier_uniform_(weight)
        nn.init.xavier_uniform_(weight_backward_B, gain=math.sqrt((1+1/expand_ratio)/2/density))
        mask_B = torch.rand_like(weight_backward_B) < density
        weight_backward_B *= mask_B
        weight_backward_R = weight_backward_B.T @ weight
        mask_R = torch.rand_like(weight_backward_R) < density
        weight_backward_R *= mask_R
        angle = compute_matrix_angle(weight, weight_backward_B @ weight_backward_R)
        angles.append(angle)
        eigenvalues = torch.real(torch.linalg.eig(
            weight_backward_B @ weight_backward_B.T
        )[0])
        BB_ev_list.append(eigenvalues)
    BB_ev_mean, BB_ev_std = torch.mean(torch.stack(BB_ev_list)), torch.std(torch.stack(BB_ev_list))
    angle_mean, angle_std = np.mean(angles), np.std(angles)
    angle_means.append(angle_mean)
    angle_stds.append(angle_std)
    BB_ev_means.append(BB_ev_mean)
    BB_ev_stds.append(BB_ev_std)
    print(f'expand_ratio={expand_ratio}, mean={angle_mean}, std={angle_std}')
angle_means = np.array(angle_means)
angle_stds = np.array(angle_stds)
BB_ev_means = np.array(BB_ev_means)
BB_ev_stds = np.array(BB_ev_stds)

fig, ax = plot_start()
plt.xscale('log')
plt.plot(ratios, BB_ev_means, label='PFA')
plt.plot(ratios, np.ones_like(ratios), '--', label='PFA-o')
plt.fill_between(ratios, BB_ev_means-BB_ev_stds, BB_ev_means+BB_ev_stds, alpha=0.6)
plt.xlabel('Expansion ratio')
plt.ylabel('Eigenvalues of $B^TB$')
plt.legend()
plt.xlim([1, 100])
plt.xticks([1, 10, 100], [ '1', '10', '100'])
plt.savefig(f'figures/eigenvalues_of_BTB_vs_ratio_density{density}.pdf', bbox_inches='tight')
plt.show()

fig, ax = plot_start()
plt.xscale('log')
plt.plot(ratios, angle_means, label='PFA')
plt.fill_between(ratios, angle_means-angle_stds, angle_means+angle_stds, alpha=0.6)
plt.plot(ratios, np.zeros_like(ratios), '--', label='PFA-o')
plt.xlabel('Expansion ratio')
plt.ylabel('Weight alignment ($^{\circ}$)')
plt.ylim([-2, 90])
plt.xlim([1, 100])
plt.xticks([1, 10, 100], ['1', '10', '100'])
plt.yticks([0, 30, 60, 90])
plt.legend()
plt.savefig(f'figures/weight_alignment_vs_ratio_density{density}.pdf', bbox_inches='tight')
plt.show()
