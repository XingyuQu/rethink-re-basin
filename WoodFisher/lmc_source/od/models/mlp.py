'''MLP in PyTorch.'''
import torch.nn as nn
from typing import List
import math

from .utils import create_linear_layer
from ..layers import ODLinear

__all__ = ["mlp"]


class ODMlp(nn.Module):
    def __init__(self, in_channels, with_od=False):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer0 = create_linear_layer(with_od, 784 * in_channels, 120)
        self.relu0 = nn.ReLU()
        self.layer1 = create_linear_layer(with_od, 120, 84)
        self.relu1 = nn.ReLU()
        self.layer2 = create_linear_layer(with_od, 84, 10, od_layer=False)

    def forward(self, input, sampler=None):
        out = self.flatten(input)
        if sampler is None:
            out = self.relu0(self.layer0(out))
            out = self.relu1(self.layer1(out))
            out = self.layer2(out)
        else:
            out = self.relu0(self.layer0(out, sampler()))
            out = self.relu1(self.layer1(out, sampler()))
            out = self.layer2(out, None)
        return out

    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def record(self, intermediate_layers: List[str]):
        """Record the output of intermediate layers."""
        self.intermediate_layers = intermediate_layers
        self.fhooks = []
        self.selected_out = {}
        self.params = dict(self.named_modules())
        for layer_name in self.intermediate_layers:
            fhook = self.params[layer_name].register_forward_hook(
                self.forward_hook(layer_name))
            self.fhooks.append(fhook)
        return self

    def stop_record(self):
        """Stop recording the output of intermediate layers."""
        # Remove hooks
        for fhook in self.fhooks:
            fhook.remove()
        del self.intermediate_layers, self.fhooks, self.selected_out
        return self


# https://github.com/samuela/git-re-basin/blob/main/src/mnist_mlp_train.py
class MnistMlp(nn.Module):
    def __init__(self, in_channels, with_od=False,
                 mnist_mlp_init_ratio=None):
        super(MnistMlp, self).__init__()
        self.flatten = nn.Flatten()
        input_size = 32 * 32 * in_channels if in_channels == 3 else\
            28 * 28
        self.layer0 = create_linear_layer(with_od, input_size, 512)
        self.relu0 = nn.ReLU()
        self.layer1 = create_linear_layer(with_od, 512, 512)
        self.relu1 = nn.ReLU()
        self.layer2 = create_linear_layer(with_od, 512, 512)
        self.relu2 = nn.ReLU()
        self.layer3 = create_linear_layer(with_od, 512, 10, od_layer=False)
        if mnist_mlp_init_ratio:
            for m in self.modules():
                if isinstance(m, (nn.Linear, ODLinear)):
                    # weights
                    F_in = m.weight.size(1)
                    std = math.sqrt(1 / F_in) * mnist_mlp_init_ratio
                    nn.init.uniform_(m.weight, -std, std)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, input, sampler=None, freeze=False):
        out = self.flatten(input)
        if sampler is None:
            out = self.relu0(self.layer0(out))
            out = self.relu1(self.layer1(out))
            out = self.relu2(self.layer2(out))
            out = self.layer3(out)
        else:
            out = self.relu0(self.layer0(out, sampler()))
            out = self.relu1(self.layer1(out, sampler()))
            out = self.relu2(self.layer2(out, sampler()))
            out = self.layer3(out, None)
        return out

    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def record(self, intermediate_layers: List[str]):
        """Record the output of intermediate layers."""
        self.intermediate_layers = intermediate_layers
        self.fhooks = []
        self.selected_out = {}
        self.params = dict(self.named_modules())
        for layer_name in self.intermediate_layers:
            fhook = self.params[layer_name].register_forward_hook(
                self.forward_hook(layer_name))
            self.fhooks.append(fhook)
        return self

    def stop_record(self):
        """Stop recording the output of intermediate layers."""
        # Remove hooks
        for fhook in self.fhooks:
            fhook.remove()
        del self.intermediate_layers, self.fhooks, self.selected_out
        return self


class SimpleMLP(nn.Module):
    def __init__(self, width=4, with_od=False, with_relu=False,
                 special_init=None):
        super(SimpleMLP, self).__init__()
        self.layer0 = create_linear_layer(with_od, 4, width, bias=False)
        self.layer1 = create_linear_layer(with_od, width, 4, bias=False,
                                          od_layer=False)
        self.relu0 = nn.ReLU() if with_relu else nn.Identity()
        # initialize layers
        if special_init is not None:
            for m in self.modules():
                if isinstance(m, (nn.Linear, ODLinear)):
                    if special_init == 'lecun_normal':
                        # weights
                        F_in = m.weight.size(1)
                        nn.init.normal_(m.weight, mean=0.,
                                        std=math.sqrt(1./F_in))
                        # biases
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    else:
                        raise ValueError(f'Unknown special init: \
                            {special_init}')

    def forward(self, input, sampler=None, **kwargs):
        if sampler is None:
            out = self.layer0(input)
            out = self.relu0(out)
            out = self.layer1(out)
        else:
            out = self.layer0(input, sampler())
            out = self.relu0(out)
            out = self.layer1(out, None)
        return out

    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def record(self, intermediate_layers: List[str]):
        """Record the output of intermediate layers."""
        self.intermediate_layers = intermediate_layers
        self.fhooks = []
        self.selected_out = {}
        self.params = dict(self.named_modules())
        for layer_name in self.intermediate_layers:
            fhook = self.params[layer_name].register_forward_hook(
                self.forward_hook(layer_name))
            self.fhooks.append(fhook)
        return self

    def stop_record(self):
        """Stop recording the output of intermediate layers."""
        # Remove hooks
        for fhook in self.fhooks:
            fhook.remove()
        del self.intermediate_layers, self.fhooks, self.selected_out
        return self


def mlp(in_channels, with_od=False):
    return ODMlp(in_channels=in_channels,
                 with_od=with_od)


def mnist_mlp(in_channels, with_od=False,
              mnist_mlp_init_ratio=None):
    return MnistMlp(in_channels=in_channels,
                    with_od=with_od,
                    mnist_mlp_init_ratio=mnist_mlp_init_ratio)


def simple_mlp(width=4, with_od=False, with_relu=False, special_init=None):
    return SimpleMLP(width=width, with_od=with_od, with_relu=with_relu,
                     special_init=special_init)
