"""
Licensed under the Apache License, Version 2.0
Modified by: PFA's authors
Originally created by: jsalbert (https://github.com/jsalbert/biotorch)
"""

import torch

from torch.cuda.amp import custom_fwd, custom_bwd
from torch import autograd


# Regular Backpropagation algorithm
class LinearGrad(autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(context, input, weight, bias=None):
        context.save_for_backward(input, weight, bias)
        output = torch.nn.functional.linear(input, weight, bias)
        return output

    @staticmethod
    @custom_bwd
    def backward(context, grad_output):
        input, weight, bias = context.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # Gradient input
        if context.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        # Gradient weights
        if context.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        # Gradient bias
        if bias is not None and context.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias
