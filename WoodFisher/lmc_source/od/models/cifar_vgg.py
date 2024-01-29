"""Based on:
https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
(14/04/2023)
Ordered VGG in PyTorch
"""

from typing import Union, List, Dict, Any, cast

import torch
import torch.nn as nn
import math

from .utils import create_conv2d_layer, create_linear_layer, Sequential
from ..layers import ODConv2d, ODLinear, BatchNorm2d, LayerNorm


class VGG(nn.Module):
    def __init__(self, features: nn.Module, special_init='vgg_init',
                 num_classes: int = 10, with_od: bool = False,
                 width_multiplier=1,
                 bias=True) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AvgPool2d(2)
        in_dim = 512 * width_multiplier
        self.classifier = Sequential(
            create_linear_layer(with_od, in_dim, num_classes, bias=bias,
                                od_layer=False),
        )

        if special_init is not None:
            for m in self.modules():
                # conv layer
                if isinstance(m, (nn.Conv2d, ODConv2d)):
                    if special_init == 'vgg_init':
                        nn.init.kaiming_normal_(
                            m.weight, mode="fan_out", nonlinearity="relu")
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif special_init == 'lecun_normal':
                        # lecun_normal init, same as flax
                        F_in = m.weight.size(1) * m.weight.size(2) *\
                            m.weight.size(3)
                        nn.init.normal_(m.weight, mean=0.,
                                        std=math.sqrt(1./F_in))
                        # zero init bias, same as flax
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                # linear layer
                elif isinstance(m, (nn.Linear, ODLinear)):
                    if special_init == 'vgg_init':
                        nn.init.normal_(m.weight, 0, 0.01)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif special_init == 'lecun_normal':
                        # lecun_normal init, same as flax
                        F_in = m.weight.size(1)
                        nn.init.normal_(m.weight, mean=0.,
                                        std=math.sqrt(1./F_in))
                        # zero init bias, same as flax
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                # bn layer
                elif isinstance(m, BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                # ln layer
                elif isinstance(m, LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, sampler=None) -> torch.Tensor:
        if sampler is None:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        else:
            x = self.features(x, sampler)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x, sampler)
        return x

    def forward_hook(self, layer_name, pre_act=False):
        def hook(module, input, output):
            self.selected_out[layer_name] = input[0] if pre_act else output
        return hook

    def record(self, intermediate_layers: List[str], pre_act=False):
        """Record the output of intermediate layers."""
        self.intermediate_layers = intermediate_layers
        self.fhooks = []
        self.selected_out = {}
        self.params = dict(self.named_modules())
        for layer_name in self.intermediate_layers:
            fhook = self.params[layer_name].register_forward_hook(
                self.forward_hook(layer_name, pre_act))
            self.fhooks.append(fhook)
        return self

    def stop_record(self):
        """Stop recording the output of intermediate layers."""
        # Remove hooks
        for fhook in self.fhooks:
            fhook.remove()
        del self.intermediate_layers, self.fhooks, self.selected_out
        return self


def make_layers(cfg: List[Union[str, int]],
                batch_norm: bool = False,
                layer_norm: bool = False,
                with_od: bool = False,
                width_multiplier=1,
                bias=True) -> Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    in_dim = 32
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            in_dim /= 2
            in_dim = int(in_dim)
        else:
            v = cast(int, v) * width_multiplier
            conv2d = create_conv2d_layer(
                with_od, in_channels, v, kernel_size=3, padding=1,
                bias=bias)
            if batch_norm:
                layers += [conv2d, BatchNorm2d(v), nn.ReLU()]
            elif layer_norm:
                layers += [conv2d, LayerNorm(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512,
          "M", 512, 512],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512,
          "M", 512, 512, 512],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256,
          "M", 512, 512, 512, 512, "M", 512, 512, 512, 512],
}


def _vgg(cfg: str, batch_norm: bool, layer_norm: bool,
         with_od: bool, width_multiplier=1, bias=True, **kwargs: Any) -> VGG:
    # kwargs: num_classes, special_init
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm,
                            layer_norm=layer_norm, with_od=with_od,
                            width_multiplier=width_multiplier,
                            bias=bias),
                with_od=with_od, width_multiplier=width_multiplier,
                bias=bias, **kwargs)
    return model


def cifar_vgg11(with_od=False, width_multiplier=1, **kwargs) -> VGG:
    return _vgg("A", batch_norm=False, layer_norm=False,
                with_od=with_od, width_multiplier=width_multiplier,
                **kwargs)


def cifar_vgg11_nobias(with_od=False, width_multiplier=1, **kwargs) -> VGG:
    return _vgg("A", batch_norm=False, layer_norm=False,
                with_od=with_od, width_multiplier=width_multiplier,
                bias=False, **kwargs)


def cifar_vgg11_bn(with_od=False, width_multiplier=1, **kwargs: Any) -> VGG:
    return _vgg("A", batch_norm=True, layer_norm=False,
                with_od=with_od, width_multiplier=width_multiplier,
                **kwargs)


def cifar_vgg11_ln(with_od=False, width_multiplier=1, **kwargs: Any) -> VGG:
    return _vgg("A", batch_norm=False, layer_norm=True,
                with_od=with_od, width_multiplier=width_multiplier,
                **kwargs)


def cifar_vgg13(with_od=False, width_multiplier=1, **kwargs: Any) -> VGG:
    return _vgg("B", batch_norm=False, layer_norm=False,
                with_od=with_od, width_multiplier=width_multiplier,
                **kwargs)


def cifar_vgg13_bn(with_od=False, width_multiplier=1, **kwargs: Any) -> VGG:
    return _vgg("B", batch_norm=True, layer_norm=False,
                with_od=with_od, width_multiplier=width_multiplier,
                **kwargs)


def cifar_vgg13_ln(with_od=False, width_multiplier=1, **kwargs: Any) -> VGG:
    return _vgg("B", batch_norm=False, layer_norm=True,
                with_od=with_od, width_multiplier=width_multiplier,
                **kwargs)


def cifar_vgg16(with_od=False, width_multiplier=1, **kwargs: Any) -> VGG:
    return _vgg("D", batch_norm=False, layer_norm=False,
                with_od=with_od, width_multiplier=width_multiplier,
                **kwargs)


def cifar_vgg16_nobias(with_od=False, width_multiplier=1, **kwargs) -> VGG:
    return _vgg("D", batch_norm=False, layer_norm=False,
                with_od=with_od, width_multiplier=width_multiplier,
                bias=False, **kwargs)


def cifar_vgg16_bn(with_od=False, width_multiplier=1, **kwargs: Any) -> VGG:
    return _vgg("D", batch_norm=True, layer_norm=False,
                with_od=with_od, width_multiplier=width_multiplier,
                **kwargs)


def cifar_vgg16_ln(with_od=False, width_multiplier=1, **kwargs: Any) -> VGG:
    return _vgg("D", batch_norm=False, layer_norm=True,
                with_od=with_od, width_multiplier=width_multiplier,
                **kwargs)


def cifar_vgg19(with_od=False, width_multiplier=1, **kwargs: Any) -> VGG:
    return _vgg("E", batch_norm=False, layer_norm=False,
                with_od=with_od, width_multiplier=width_multiplier,
                **kwargs)


def cifar_vgg19_bn(with_od=False, width_multiplier=1, **kwargs: Any) -> VGG:
    return _vgg("E", batch_norm=True, layer_norm=False,
                with_od=with_od, width_multiplier=width_multiplier,
                **kwargs)


def cifar_vgg19_ln(with_od=False, width_multiplier=1, **kwargs: Any) -> VGG:
    return _vgg("E", batch_norm=False, layer_norm=True,
                with_od=with_od, width_multiplier=width_multiplier,
                **kwargs)
