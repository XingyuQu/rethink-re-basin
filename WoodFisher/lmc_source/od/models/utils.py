from torch import nn

from ..layers import ODConv2d, ODLinear


def create_linear_layer(with_od=False, *args, od_layer=True, **kwargs):
    if with_od:
        return ODLinear(*args, od_layer=od_layer, **kwargs)
    else:
        return nn.Linear(*args, **kwargs)


def create_conv2d_layer(with_od=False, *args, od_layer=True, **kwargs):
    if with_od:
        return ODConv2d(*args, od_layer=od_layer, **kwargs)
    else:
        return nn.Conv2d(*args, **kwargs)


class Sequential(nn.Sequential):

    def __init__(self, *layers):
        super(Sequential, self).__init__(*layers)

    def forward(self, input, sampler=None):
        if sampler is None:
            for module in self:
                input = module(input)
        else:
            for module in self:
                if getattr(module, 'od_layer', False):
                    input = module(input, sampler())
                else:
                    input = module(input)
        return input
