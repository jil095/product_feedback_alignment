"""
Licensed under the Apache License, Version 2.0
Created by: PFA's authors
"""

from torch import autograd
from torch.cuda.amp import custom_fwd, custom_bwd

class LinearGrad(autograd.Function):
    """
    Autograd Function that Does a backward pass using the weight_backward matrix of the layer
    """
    @staticmethod
    @custom_fwd
    # Same as reference linear function, but with additional weight tensor for backward
    def forward(context, input, weight, weight_backward_B, weight_backward_R, bias=None, bias_backward=None):
        context.save_for_backward(input, weight, weight_backward_B, weight_backward_R, bias, bias_backward)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    @custom_bwd
    def backward(context, grad_output_B):
        input, weight, weight_backward_B, weight_backward_R, bias, bias_backward = context.saved_tensors
        grad_input = grad_weight = grad_weight_backward_B = grad_weight_backward_R = grad_bias = grad_bias_backward = None
        grad_output_R = grad_output_B.mm(weight_backward_B)

        # Gradient input
        if context.needs_input_grad[0]:
            # Use the weight_backward matrix to compute the gradient
            grad_input = grad_output_R.mm(weight_backward_R)
        # Gradient weights
        if context.needs_input_grad[1]:
            grad_weight = grad_output_B.t().mm(input)
        # Gradient weights backward B
        assert not context.needs_input_grad[2]
        # Gradient weights backward R
        if context.needs_input_grad[3]:
            grad_weight_backward_R = grad_output_R.t().mm(input)
        # Gradient bias
        if bias is not None and context.needs_input_grad[4]:
            grad_bias = grad_output_B.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_weight_backward_B, grad_weight_backward_R, grad_bias, grad_bias_backward
