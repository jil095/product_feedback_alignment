"""
Licensed under the Apache License, Version 2.0
Modified by: PFA's authors
Originally created by: jsalbert (https://github.com/jsalbert/biotorch)
"""
import math
import torch
import torch.nn as nn


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, layer_config: dict = None) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)
        self.layer_config = layer_config

        if self.layer_config is None:
            self.layer_config = {
                "type": "backpropagation"
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
        self.w_conn_density = self.options["w_conn_density"] if "w_conn_density" in self.options else 1
        self.init_parameters()

    def init_parameters(self) -> None:
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
        # Xavier initialization
        if self.init == "xavier":
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.constant_(self.bias, 0)
        # Pytorch Default (Kaiming)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

        self.masked_weights()

    def masked_weights(self):
        if self.w_conn_density < 1:
            if not hasattr(self, "W_mask"):
                self.W_mask = torch.rand_like(self.weight) < self.w_conn_density
            print("Masked weights with conn_density: ", 'w:', self.w_conn_density)
            if self.W_mask.device != self.weight.device:
                self.W_mask = self.W_mask.to(self.weight.device, dtype=self.weight.dtype)
            with torch.no_grad():
                self.weight.data *= self.W_mask

    def forward(self, x):
        self.masked_weights()
        return super().forward(x)