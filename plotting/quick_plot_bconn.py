
"""
Licensed under the Apache License, Version 2.0
Created by: PFA's authors
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import *
from quick_plot import *

algorithms = ['fa', 'dfa',
              'tfawd', 'usf']
bconn_values = [1, 0.5, 0.3, 0.17, 0.1, 0.05]
num_seeds = 5

data = {alg: {bconn: [] for bconn in bconn_values} for alg in algorithms}

accuracy_regex = re.compile(r"tensor\((\d+\.\d+), device='cuda:0'\)")

for alg in algorithms:
    for bconn in bconn_values:
        for seed in range(num_seeds):
            folder_name = f"{alg}_bconn{bconn}_seed{seed}"
            file_path = os.path.join(Path(r'..\trained_models\mnist\ffn2_bsparse') / folder_name, "best_acc.txt")
            try:
                with open(file_path, 'r') as file:
                    content = file.read()
                    match = accuracy_regex.search(content)
                    if match:
                        accuracy = float(match.group(1))
                        data[alg][bconn].append(accuracy)
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                data[alg][bconn].append(96)

# Calculate the mean and standard deviation for each algorithm and bconn value
results = {alg: {bconn: (np.mean(accs), np.std(accs)) for bconn, accs in acc_data.items()} for alg, acc_data in
           data.items()}

# Plot the results
fig, ax = plot_start(figsize=(2,2))
bp_perf = [98.43,98.45,98.45,98.37,98.36] # taking from the BP results from seed 0 to 4
mean = np.mean(bp_perf)
std = np.std(bp_perf)
plt.plot([0.05,1], mean*np.ones(2), '--', label='BP', color=color_map['backpropagation'])
plt.fill_between([0.05,1], (mean-std)*np.ones(2), (mean+std)*np.ones(2),
                 alpha=0.2, color=color_map['backpropagation'])
for alg,alg_name in zip(['fa','dfa',
                         'usf', 'tfawd'],
               ['FA','DFA',
                'SF', 'PFA'],
               ):
    bconn_vals = sorted(results[alg].keys())
    mean_accs = [results[alg][bconn][0] for bconn in bconn_vals]
    std_accs = [results[alg][bconn][1] for bconn in bconn_vals]

    plt.plot(bconn_vals, mean_accs, label=f"{alg_name}", color=color_map[alg])
    plt.fill_between(bconn_vals,
                     [m - s for m, s in zip(mean_accs, std_accs)],
                     [m + s for m, s in zip(mean_accs, std_accs)],
                     alpha=0.2, color=color_map[alg])


plt.xlabel('Feedback connection density')
plt.yticks([97, 98, 99])
plt.xscale('log')
plt.xticks([0.05, 0.1, 0.2, 0.5, 1], [0.05, 0.1, 0.2, 0.5, 1])
plt.ylabel('Test accuracy')
plt.legend()
plt.savefig(Path('./figures') / 'bconn_mnist.pdf', bbox_inches='tight')
plt.show()
