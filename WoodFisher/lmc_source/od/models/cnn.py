from typing import Union, List, Dict, Any, cast

import torch
import torch.nn as nn
import math


def conv3x3(in_planes, planes, bias=False):
    return nn.Conv2d(in_planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=bias)


def avg2x2():
    return nn.AvgPool2d(2)


def SimpleCNN(dataset, bias=False, bn=False):
    """
    To see how arch_str is turned into a CNN, check config.py and the doc string
    for cnn_arch.
    """

    if "mnist" in dataset:
        input_channels = 1
        width = 28
    elif "cifar" in dataset:
        input_channels = 3
        width = 32
    elif "svhn" in dataset:
        input_channels = 3
        width = 32
    else:
        print("Dataset not supported.")
        exit()
    arch_str = '128-128-avg-256-256'
    arch_str = arch_str.split('-')
    mod_list = []

    out_c = input_channels
    for a in arch_str:
        if a == 'avg':
            mod_list.append(avg2x2())
            width //= 2
        else:
            in_c = out_c
            out_c = int(a)
            mod_list.append(conv3x3(in_c, out_c))
            if bn:
                mod_list.append(nn.BatchNorm2d(out_c))
            mod_list.append(nn.ReLU())
    mod_list.append(nn.AvgPool2d(width))
    mod_list.append(nn.Flatten())
    mod_list.append(nn.Linear(out_c, 10, bias=bias))
    return nn.Sequential(*mod_list)
