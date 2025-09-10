"""
Licensed under the Apache License, Version 2.0
Created by: PFA's authors
"""
import math
import torch
import torch.nn as nn
from torch import Tensor

from biotorch.autograd.tfa.linear import LinearGrad
from biotorch.layers.metrics import compute_matrix_angle


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, layer_config: dict = None) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)
        self.layer_config = layer_config

        if self.layer_config is None:
            self.layer_config = {
                "type": "tfa"
            }

        if "options" not in self.layer_config:
            self.layer_config["options"] = {
                "constrain_weights": False,
                "gradient_clip": False,
                "init": "xavier"
            }

        self.options = self.layer_config["options"]
        self.type = self.layer_config["type"]
        self.init = self.options["init"]
        self.expand_ratio = self.options["expand_ratio"]
        self.w_conn_density = self.options["w_conn_density"] if "w_conn_density" in self.options else 1
        self.b_conn_density = self.options["b_conn_density"] if "b_conn_density" in self.options else 1
        if isinstance(self.expand_ratio, str):
            if self.expand_ratio == "id":
                self.expand_ratio = 1
                self.B_type = 'id'
            elif self.expand_ratio == "ortho":
                self.expand_ratio = 1
                self.B_type = 'ortho'
            else:
                raise ValueError("Invalid expand ratio")
        else:
            self.B_type = 'rand'

        N_out, N_in = self.weight.size()
        N_mid = N_out * self.expand_ratio
        self.weight_backward_R = nn.Parameter(torch.Tensor(N_mid, N_in), requires_grad=True)

        if self.B_type == 'id':
            self.weight_backward_B = nn.Parameter(torch.eye(N_out, N_mid), requires_grad=False)
        elif self.B_type == 'ortho':
            self.weight_backward_B = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(N_out, N_mid)), requires_grad=False)
        else:
            self.weight_backward_B = nn.Parameter(torch.Tensor(N_out, N_mid), requires_grad=False)

        if self.bias is not None:
            N_b = self.bias.size()
            assert len(N_b) == 1
            self.bias_backward = nn.Parameter(torch.Tensor(N_b[0]*self.expand_ratio), requires_grad=False)
        else:
            self.register_parameter("bias", None)
            self.bias_backward = None

        self.init_parameters()

        if "constrain_weights" in self.options and self.options["constrain_weights"]:
            self.norm_initial_weights = torch.linalg.norm(self.weight)

        self.alignment = 0
        self.weight_ratio = 0

        if "gradient_clip" in self.options and self.options["gradient_clip"]:
            self.register_backward_hook(self.gradient_clip)

    def init_parameters(self) -> None:
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
        # Xavier initialization
        if self.init == "xavier":
            nn.init.xavier_uniform_(self.weight)
            nn.init.xavier_uniform_(self.weight_backward_R)
            if self.B_type == 'rand':
                nn.init.xavier_uniform_(self.weight_backward_B,gain=math.sqrt((1+1/self.expand_ratio)/2/self.b_conn_density))
            # Scaling factor is the standard deviation of xavier init.
            self.scaling_factor = math.sqrt(2.0 / float(fan_in + fan_out))
            if self.bias is not None:
                nn.init.constant_(self.bias, 0)
                nn.init.constant_(self.bias_backward, 0)
        # Pytorch Default (Kaiming)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight_backward_R, a=math.sqrt(5))
            if self.B_type == 'rand':
                nn.init.kaiming_uniform_(self.weight_backward_B, a=math.sqrt(5))
            # Scaling factor is the standard deviation of Kaiming init.
            self.scaling_factor = 1 / math.sqrt(3 * fan_in)
            if self.bias is not None:
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
                nn.init.uniform_(self.bias_backward, -bound, bound)


        self.masked_weights()

    def masked_weights(self):
        if self.b_conn_density < 1 or self.w_conn_density < 1:
            if not hasattr(self, "W_mask"):
                self.W_mask = torch.rand_like(self.weight) < self.w_conn_density
                self.B_mask = torch.rand_like(self.weight_backward_B) < self.b_conn_density
                self.R_mask = torch.rand_like(self.weight_backward_R) < self.b_conn_density
                print("Masked weights with conn_density: ", 'w:', self.w_conn_density, 'b:', self.b_conn_density)
                
            if self.W_mask.device != self.weight.device:
                self.W_mask = self.W_mask.to(self.weight.device, dtype=self.weight.dtype)
                self.B_mask = self.B_mask.to(self.weight_backward_B.device, dtype=self.weight_backward_B.dtype)
                self.R_mask = self.R_mask.to(self.weight_backward_R.device, dtype=self.weight_backward_R.dtype)
            with torch.no_grad():
                self.weight.data *= self.W_mask
                self.weight_backward_B.data *= self.B_mask
                self.weight_backward_R.data *= self.R_mask

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            # Based on "Feedback alignment in deep convolutional networks" (https://arxiv.org/pdf/1812.06488.pdf)
            # Constrain weight magnitude
            if "constrain_weights" in self.options and self.options["constrain_weights"]:
                self.weight = torch.nn.Parameter(
                    self.weight * self.norm_initial_weights / torch.linalg.norm(self.weight))

        self.masked_weights()
        return LinearGrad.apply(x, self.weight,
                                self.weight_backward_B,
                                self.weight_backward_R,
                                self.bias, self.bias_backward)

    def compute_alignment(self):
        with torch.no_grad():
            self.masked_weights()
            weight_backward = torch.matmul(self.weight_backward_B, self.weight_backward_R)
            self.alignment = compute_matrix_angle(weight_backward, self.weight)
        return self.alignment

    def compute_weight_ratio(self):
        with torch.no_grad():
            self.masked_weights()
            weight_backward = torch.matmul(self.weight_backward_B, self.weight_backward_R)
            self.weight_diff = torch.linalg.norm(weight_backward) / torch.linalg.norm(self.weight)
        return self.weight_diff

    @staticmethod
    def gradient_clip(module, grad_input, grad_output):
        grad_input = list(grad_input)
        for i in range(len(grad_input)):
            if grad_input[i] is not None:
                grad_input[i] = torch.clamp(grad_input[i], -1, 1)
        return tuple(grad_input)
