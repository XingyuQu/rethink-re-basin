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

__all__ = [
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
]


class VGG(nn.Module):
    def __init__(self, features: nn.Module, special_init='vgg_init',
                 num_classes: int = 1000,
                 dropout: float = 0.5, with_od: bool = False,
                 with_dp: bool = True, git_rebasin_model=False) -> None:
        super().__init__()
        self.features = features
        self.git_rebasin_model = git_rebasin_model
        if not self.git_rebasin_model:
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        in_dim = 512 * 1 * 1 if git_rebasin_model else 512 * 7 * 7
        hidden_dim = 512 if git_rebasin_model else 4096
        self.classifier = Sequential(
            create_linear_layer(with_od, in_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(p=dropout) if with_dp else nn.Identity(),
            create_linear_layer(with_od, hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(p=dropout) if with_dp else nn.Identity(),
            create_linear_layer(with_od, hidden_dim, num_classes,
                                od_layer=False),
        )
        if special_init is not None:
            for m in self.modules():
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
                elif isinstance(m, BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.GroupNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.Linear, ODLinear)):
                    if special_init == 'vgg_init':
                        nn.init.normal_(m.weight, 0, 0.01)
                        nn.init.constant_(m.bias, 0)
                    elif special_init == 'lecun_normal':
                        # lecun_normal init, same as flax
                        F_in = m.weight.size(1)
                        nn.init.normal_(m.weight, mean=0.,
                                        std=math.sqrt(1./F_in))
                        # zero init bias, same as flax
                        nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, sampler=None, freeze=False)\
            -> torch.Tensor:
        if sampler is None:
            x = self.features(x)
            if not self.git_rebasin_model:
                x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        else:
            # if freeze:
            #     params = dict(self.named_modules())
            #     for id in range(0, sampler.n_layers):
            #         p_last = sampler.samples_last[id-1] if id > 0 else\
            #             None
            #         p = sampler.samples_last[id]
            #         if p_last is None and p is None:
            #             name = sampler.names[id]
            #             params[name].requires_grad_(False)
            #     if sampler.samples_last[-1] is None:
            #         self.classifier[-1].requires_grad_(False)

            x = self.features(x, sampler)
            if not self.git_rebasin_model:
                x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x, sampler)
            # if freeze:
            #     self.requires_grad_(True)
        return x

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
        return 'classifier.6.weight'


def make_layers(cfg: List[Union[str, int]],
                batch_norm: bool = False,
                layer_norm: bool = False,
                group_norm: bool = False,
                with_od: bool = False) -> Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    in_dim = 32
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            in_dim /= 2
            in_dim = int(in_dim)
        else:
            v = cast(int, v)
            conv2d = create_conv2d_layer(
                with_od, in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, BatchNorm2d(v), nn.ReLU()]
            elif layer_norm:
                layers += [conv2d, LayerNorm(v), nn.ReLU()]
            elif group_norm:
                layers += [conv2d, nn.GroupNorm(num_groups=8, num_channels=v),
                           nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512,
          "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512,
          "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256,
          "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(cfg: str, batch_norm: bool, layer_norm: bool,
         with_od: bool, group_norm=False, **kwargs: Any) -> VGG:
    # kwargs: num_classes, with_dp, special_init, git_rebasin_model
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm,
                            layer_norm=layer_norm, with_od=with_od,
                            group_norm=group_norm),
                with_od=with_od, **kwargs)
    return model


def vgg11(with_od=False, **kwargs) -> VGG:
    """VGG-11 from `Very Deep Convolutional Networks for
        Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    """
    return _vgg("A", batch_norm=False, layer_norm=False,
                with_od=with_od, **kwargs)


def vgg11_bn(with_od=False, **kwargs: Any) -> VGG:
    """VGG-11-BN from `Very Deep Convolutional Networks for
        Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    """

    return _vgg("A", batch_norm=True, layer_norm=False,
                with_od=with_od, **kwargs)


def vgg11_ln(with_od=False, **kwargs: Any) -> VGG:
    """VGG-11-LN
    """
    return _vgg("A", batch_norm=False, layer_norm=True,
                with_od=with_od, **kwargs)


def vgg11_gn(with_od=False, **kwargs: Any) -> VGG:
    return _vgg("A", batch_norm=False, layer_norm=False,
                group_norm=True, with_od=with_od, **kwargs)


def vgg13(with_od=False, **kwargs: Any) -> VGG:
    """VGG-13 from `Very Deep Convolutional Networks for
        Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    """
    return _vgg("B", batch_norm=False, layer_norm=False,
                with_od=with_od, **kwargs)


def vgg13_bn(with_od=False, **kwargs: Any) -> VGG:
    """VGG-13-BN from `Very Deep Convolutional Networks for
        Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    """
    return _vgg("B", batch_norm=True, layer_norm=False,
                with_od=with_od, **kwargs)


def vgg13_ln(with_od=False, **kwargs: Any) -> VGG:
    """VGG-13-LN
    """
    return _vgg("B", batch_norm=False, layer_norm=True,
                with_od=with_od, **kwargs)


def vgg16(with_od=False, **kwargs: Any) -> VGG:
    """VGG-16 from `Very Deep Convolutional Networks for
        Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    """
    return _vgg("D", batch_norm=False, layer_norm=False,
                with_od=with_od, **kwargs)


def vgg16_bn(with_od=False, **kwargs: Any) -> VGG:
    """VGG-16-BN from `Very Deep Convolutional Networks for
        Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    """
    return _vgg("D", batch_norm=True, layer_norm=False,
                with_od=with_od, **kwargs)


def vgg16_ln(with_od=False, **kwargs: Any) -> VGG:
    """VGG-16-BN from `Very Deep Convolutional Networks for
        Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    """
    return _vgg("D", batch_norm=False, layer_norm=True,
                with_od=with_od, **kwargs)


def vgg19(with_od=False, **kwargs: Any) -> VGG:
    """VGG-19 from `Very Deep Convolutional Networks for
        Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    """
    return _vgg("E", batch_norm=False, layer_norm=False,
                with_od=with_od, **kwargs)


def vgg19_bn(with_od=False, **kwargs: Any) -> VGG:
    """VGG-19_BN from `Very Deep Convolutional Networks for
        Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    """
    return _vgg("E", batch_norm=True, layer_norm=False,
                with_od=with_od, **kwargs)


def vgg19_ln(with_od=False, **kwargs: Any) -> VGG:
    """VGG-19_LN
    """
    return _vgg("E", batch_norm=False, layer_norm=True,
                with_od=with_od, **kwargs)
