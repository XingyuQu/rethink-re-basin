'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math

from .utils import create_linear_layer, create_conv2d_layer
from ..layers import ODLinear, ODConv2d

__all__ = ["lenet"]


class ODLeNet(nn.Module):
    def __init__(self, in_channels=1, with_od=False, special_init=None):
        super().__init__()
        self.conv0 = create_conv2d_layer(with_od, in_channels, 6, 5)
        self.relu0 = nn.ReLU()
        self.conv1 = create_conv2d_layer(with_od, 6, 16, 5)
        self.relu1 = nn.ReLU()
        if in_channels == 1:
            self.fc0 = create_linear_layer(with_od, 256, 120)
        elif in_channels == 3:
            self.fc0 = create_linear_layer(with_od, 400, 120)
        self.relu2 = nn.ReLU()
        self.fc1 = create_linear_layer(with_od, 120, 84)
        self.relu3 = nn.ReLU()
        self.fc2 = create_linear_layer(with_od, 84, 10, od_layer=False)
        # init layers
        if special_init is not None:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, ODConv2d)):
                    # weight
                    if special_init == 'lecun_normal':
                        F_in = m.weight.size(1) * m.weight.size(2) * \
                            m.weight.size(3)
                        nn.init.normal_(m.weight, mean=0.,
                                        std=math.sqrt(1./F_in))

                    elif special_init == 'xavier_normal':
                        nn.init.xavier_normal_(m.weight)
                    elif special_init == 'orthogonal':
                        nn.init.orthogonal_(m.weight)
                    elif special_init == 'vgg_init':
                        nn.init.kaiming_normal_(
                            m.weight, mode="fan_out", nonlinearity="relu")
                    else:
                        raise NotImplementedError
                    # bias = 0
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.Linear, ODLinear)):
                    if special_init == 'lecun_normal':
                        F_in = m.weight.size(1)
                        nn.init.normal_(m.weight, mean=0.,
                                        std=math.sqrt(1./F_in))
                    elif special_init == 'xavier_normal':
                        nn.init.xavier_normal_(m.weight,)
                    elif special_init == 'orthogonal':
                        nn.init.orthogonal_(m.weight)
                    elif special_init == 'vgg_init':
                        nn.init.normal_(m.weight, 0, 0.01)
                    else:
                        raise NotImplementedError
                    # bias = 0
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, input, sampler=None, freeze=False):
        # TODO: freeze
        if sampler is None:
            out = self.relu0(self.conv0(input))
            out = F.max_pool2d(out, 2)
            out = self.relu1(self.conv1(out))
            out = F.max_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out = self.relu2(self.fc0(out))
            out = self.relu3(self.fc1(out))
            out = self.fc2(out)
        else:
            out = self.relu0(self.conv0(input, sampler()))
            out = F.max_pool2d(out, 2)
            out = self.relu1(self.conv1(out, sampler()))
            out = F.max_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out = self.relu2(self.fc0(out, sampler()))
            out = self.relu3(self.fc1(out, sampler()))
            out = self.fc2(out, None)
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

    @property
    def prunable_layer_names(self):
        return [name + '.weight' for name, module in
                self.named_modules() if
                isinstance(module, (nn.Conv2d, ODConv2d)) or
                isinstance(module, (nn.Linear, ODLinear))]

    @property
    def output_layer(self):
        return 'fc2.weight'


class LeNet_T(nn.Module):
    def __init__(self, in_channels=1, with_od=False):
        super().__init__()

        ouvolume = 800 if in_channels == 1 else 1250
        self.conv0 = create_conv2d_layer(with_od, in_channels, 20, 5)
        self.relu0 = nn.ReLU()
        self.conv1 = create_conv2d_layer(with_od, 20, 50, 5)
        self.relu1 = nn.ReLU()
        self.fc0 = create_linear_layer(with_od, ouvolume, 500)
        self.relu2 = nn.ReLU()
        self.fc1 = create_linear_layer(with_od, 500, 10, od_layer=False)

    def forward(self, input, sampler=None, freeze=False):
        # TODO: freeze
        if sampler is None:
            out = self.relu0(self.conv0(input))
            out = F.max_pool2d(out, 2)
            out = self.relu1(self.conv1(out))
            out = F.max_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out = self.relu2(self.fc0(out))
            out = self.fc1(out)
        else:
            out = self.relu0(self.conv0(input, sampler()))
            out = F.max_pool2d(out, 2)
            out = self.relu1(self.conv1(out, sampler()))
            out = F.max_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out = self.relu2(self.fc0(out, sampler()))
            out = self.fc1(out, None)
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


def lenet(in_channels=1, with_od=False, special_init=None):
    return ODLeNet(in_channels=in_channels, with_od=with_od,
                   special_init=special_init)


def lenet_t(in_channels=1, with_od=False, special_init=None):
    return LeNet_T(in_channels=in_channels, with_od=with_od)
