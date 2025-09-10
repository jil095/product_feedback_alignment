
<h3 align="center">
    <p>Implementation of product feedback alignment algorithm based on BioTorch</p>
</h3>

---
This repository implements the product feedback alignment algorithm for the paper "Deep Learning without Weight Symmetry". 
It is forked from and edited based on [BioTorch](https://github.com/jsalbert/biotorch), which provides:
- Implementations of layers, models and biologically-motivated learning algorithms.
- A framework to train, evaluate and benchmark different biologically plausible learning algorithms in a selection of datasets. 

## Package requirements
- torch (cuda)
- numpy
- matplotlib
- tensorboard
- torchvision
- tqdm
- pandas
- einops
- tbparse
- jsonschema
- Pillow
- pyyaml
- scipy


## Methods included

- Product Feedback Alignment (PFA): proposed in this paper
- Feedback Alignment (FA): originally supported by BioTorch
- Direct Feedback Alignment (DFA): originally supported by BioTorch
- Sign Symmetry (SF): originally supported by BioTorch

## Metrics included

- Path Alignment: metric for PFA; proposed in this paper
- Path Norm Ratio: metric for PFA; proposed in this paper
- Weight Alignment: metric for FA, DFA, SF; originally supported by BioTorch
- Weight Norm Ratio: metric for FA, DFA, SF; originally supported by BioTorch


## Datasets supported

- MNIST
- CIFAR-10
- ImageNet

## Run an experiment on the command line
E.g.: python benchmark.py --config exp_configs\mnist_ffn2_sgd\tfawd_seed0.yaml

File name explanation:

"tfawd": PFA with weight decay

"tfawdo": PFA with weight decay and orthogonal initialization

"usf": uniform SF

## Key py files for PFA

- biotorch/autograd/tfa/*.py
- biotorch/layers/tfa/*.py
- biotorch/layers/tfa_constructor/*.py
- biotorch/models/tfa/*.py

## Figures
- plotting/alignment_vs_expand_ratio.py: plot Figure 3 in the paper
- plotting/quick_plot.py: plot Figure 2,4,5 in the paper (by editing exp_name in the script)

