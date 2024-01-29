"""Based on:
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
(14/04/2023)
Ordered ResNet in PyTorch
"""

from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .utils import create_conv2d_layer, create_linear_layer, Sequential
from ..layers import ODConv2d, BatchNorm2d


__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


def conv3x3(with_od: bool, in_planes: int, out_planes: int,
            stride: int = 1, groups: int = 1, dilation: int = 1
            ) -> Union[nn.Conv2d, ODConv2d]:
    """3x3 convolution with padding"""
    return create_conv2d_layer(
        with_od,
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(with_od: bool, in_planes: int, out_planes: int, stride: int = 1
            ) -> Union[nn.Conv2d, ODConv2d]:
    """1x1 convolution"""
    return create_conv2d_layer(
        with_od,
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        with_od: bool,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample
        #  the input when stride != 1
        self.conv1 = conv3x3(with_od, inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(with_od, planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, sampler) -> Tensor:
        identity = x
        if sampler is None:

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)
        else:
            out = self.conv1(x, sampler())
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out, sampler())
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x, sampler)

            out += identity
            out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling
    #  at 3x3 convolution(self.conv2)
    # while original implementation places the stride
    #  at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for
    #  image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and
    #  improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        with_od: bool,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers
        #  downsample the input when stride != 1
        self.conv1 = conv1x1(with_od, inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(with_od, width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(with_od, width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, sampler=None) -> Tensor:
        identity = x
        if sampler is None:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)
        else:
            out = self.conv1(x, sampler())
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out, sampler())
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out, sampler())
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x, sampler)

            out += identity
            out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        with_od: bool,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = create_conv2d_layer(
            with_od, 3, self.inplanes, kernel_size=7,
            stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            with_od, block, 64, layers[0])
        self.layer2 = self._make_layer(
            with_od, block, 128, layers[1], stride=2,
            dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            with_od, block, 256, layers[2], stride=2,
            dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            with_od, block, 512, layers[3], stride=2,
            dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = create_linear_layer(
            with_od, 512 * block.expansion, num_classes, od_layer=False)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, ODConv2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros,
        #  and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        #  https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                    # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)
                    # type: ignore[arg-type]

    def _make_layer(
        self,
        with_od: bool,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(with_od, self.inplanes,
                        planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                with_od, self.inplanes, planes, stride, downsample,
                self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    with_od,
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return Sequential(*layers)

    def _forward_impl(self, x: Tensor, sampler=None) -> Tensor:
        # See note [TorchScript super()]
        if sampler is None:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        else:
            x = self.conv1(x, sampler())
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for layer in layers:
                for block in layer:
                    x = block(x, sampler)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x, None)

        return x

    def forward(self, x: Tensor, sampler=None) -> Tensor:
        return self._forward_impl(x, sampler)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(**kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for
    Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.
    """

    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs: Any) -> ResNet:
    """ResNet-34 from `Deep Residual Learning for
    Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.
    """
    return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs: Any) -> ResNet:
    """ResNet-50 from `Deep Residual Learning for
    Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs: Any) -> ResNet:
    """ResNet-101 from `Deep Residual Learning for
    Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.
    """
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs: Any) -> ResNet:
    """ResNet-152 from `Deep Residual Learning for
    Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.
    """
    return _resnet(Bottleneck, [3, 8, 36, 3], **kwargs)


def resnext50_32x4d(
    **kwargs: Any
) -> ResNet:
    """ResNeXt-50 32x4d model from
    `Aggregated Residual Transformation for
    Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnext101_32x8d(
    **kwargs: Any
) -> ResNet:
    """ResNeXt-101 32x8d model from
    `Aggregated Residual Transformation for
    Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.
    """
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnext101_64x4d(
    **kwargs: Any
) -> ResNet:
    """ResNeXt-101 64x4d model from
    `Aggregated Residual Transformation for
    Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.
    """
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def wide_resnet50_2(
    **kwargs: Any
) -> ResNet:
    """Wide ResNet-50-2 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def wide_resnet101_2(
    **kwargs: Any
) -> ResNet:
    """Wide ResNet-101-2 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.
    """
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)
