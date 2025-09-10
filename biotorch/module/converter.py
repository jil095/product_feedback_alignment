"""
Licensed under the Apache License, Version 2.0
Modified by: PFA's authors
Originally created by: jsalbert (https://github.com/jsalbert/biotorch)
"""

from collections import defaultdict
from biotorch.layers.utils import convert_layer


class ModuleConverter:
    def __init__(self, mode='fa'):
        self.mode = mode

    def convert(self, module, copy_weights=True, layer_config=None, output_dim=None):
        # Compute original model layer counts
        layer_counts = self.count_layers(module)
        # Replace layers
        self.replaced_layers_counts = defaultdict(lambda: 0)
        self.recursive_depth = 0
        self._replace_layers_recursive(module, self.mode, copy_weights, layer_config, output_dim, self.replaced_layers_counts)
        # Sanity Check
        print('Module has been converted to {} mode:\n'.format(self.mode))
        if layer_config is not None:
            print('The layer configuration was: ', layer_config)
        for layer, count in self.replaced_layers_counts.items():
            if layer_counts[layer] != count:
                print('- There were originally {} {} layers and {} were converted.'.format(layer_counts[layer],
                                                                                           layer,
                                                                                           count))
            else:
                print('- All the {} {} layers were converted successfully.'.format(count, layer))

        return module

    def _replace_layers_recursive(self, module, mode, copy_weights, layer_config, output_dim, replaced_layers):
        # Go through all of module nn.module (e.g. network or layer)
        for module_name in module._modules.keys():
            # Get layer
            layer = getattr(module, module_name)
            # Convert layer
            new_layer = convert_layer(layer, mode, copy_weights, layer_config, output_dim)
            if new_layer is not None:
                replaced_layers[str(type(layer))] += 1
                setattr(module, module_name, new_layer)
                print('depth', self.recursive_depth,
                      'converting layer: ', module_name,
                      'type',str(type(layer)),
                      'index',replaced_layers[str(type(layer))]-1,
                      'shape',layer.weight.shape)

        # Iterate through immediate child modules
        self.recursive_depth += 1
        for name, child_module in module.named_children():
            self._replace_layers_recursive(child_module, mode, copy_weights, layer_config, output_dim, replaced_layers)
        self.recursive_depth -= 1

    @staticmethod
    def count_layers(module):
        layer_counts = defaultdict(lambda: 0)
        for layer in module.modules():
            layer_counts[str(type(layer))] += 1

        return layer_counts
