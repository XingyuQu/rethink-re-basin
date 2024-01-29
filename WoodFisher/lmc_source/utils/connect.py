from copy import deepcopy
import torch
import torch.nn as nn
from ..layers.batch_norm import bn_calibration_init
from torch import Tensor
from torch.nn import BatchNorm2d
from utils.masking_utils import WrappedLayer


# https://github.com/KellerJordan/REPAIR/blob/master/notebooks/Train-Merge-REPAIR-VGG11.ipynb
class TrackLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.bn = nn.BatchNorm2d(layer._layer.out_channels)

    def get_stats(self):
        return (self.bn.running_mean, self.bn.running_var.sqrt())

    def forward(self, x):
        x1 = self.layer(x)
        self.bn(x1)
        return x1


class ResetLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.bn = nn.BatchNorm2d(layer._layer.out_channels)

    def set_stats(self, goal_mean, goal_std):
        self.bn.bias.data = goal_mean
        self.bn.weight.data = goal_std

    def forward(self, x):
        x1 = self.layer(x)
        return self.bn(x1)


class RescaleLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.bn = BatchScale2d(layer._layer.out_channels)

    def set_stats(self, goal_std):
        self.bn.weight.data = goal_std

    def forward(self, x):
        x1 = self.layer(x)
        return self.bn(x1)


# adds TrackLayers around every conv layer
def make_tracked_net(net, device=None, name='vgg'):
    net1 = deepcopy(net)
    if 'vgg' in name:
        for i, layer in enumerate(net1.module.features):
            if isinstance(layer, (WrappedLayer)):
                net1.module.features[i] = TrackLayer(layer)
    elif 'resnet20' in name:
        for block_group in net1.module.segments:
            for block in block_group:
                block.conv1 = TrackLayer(block.conv1)
                block.conv2 = TrackLayer(block.conv2)
    elif 'resnet50' in name:
        for layer_id in [1, 2, 3, 4]:
            for layer in getattr(net1.module, f'layer{layer_id}'):
                layer.conv1 = TrackLayer(layer.conv1)
                layer.conv2 = TrackLayer(layer.conv2)
                layer.conv3 = TrackLayer(layer.conv3)
    else:
        raise NotImplementedError
    return net1.eval().to(device)


# adds ResetLayers around every conv layer
def make_repaired_net(net, device=None, name='vgg'):
    net1 = deepcopy(net).to(device)
    if 'vgg' in name:
        for i, layer in enumerate(net1.module.features):
            if isinstance(layer, (WrappedLayer)):
                net1.module.features[i] = ResetLayer(layer)
    elif 'resnet20' in name:
        for block_group in net1.module.segments:
            for block in block_group:
                block.conv1 = ResetLayer(block.conv1)
                block.conv2 = ResetLayer(block.conv2)
    elif 'resnet50' in name:
        for layer_id in [1, 2, 3, 4]:
            for layer in getattr(net1.module, f'layer{layer_id}'):
                layer.conv1 = ResetLayer(layer.conv1)
                layer.conv2 = ResetLayer(layer.conv2)
                layer.conv3 = ResetLayer(layer.conv3)
    else:
        raise NotImplementedError
    return net1.eval().to(device)


def make_rescale_net(net, device=None, name='vgg'):
    net1 = deepcopy(net).to(device)
    if 'vgg' in name:
        for i, layer in enumerate(net1.module.features):
            if isinstance(layer, (WrappedLayer)):
                net1.module.features[i] = RescaleLayer(layer)
    elif 'resnet20' in name:
        for block_group in net1.module.segments:
            for block in block_group:
                block.conv1 = RescaleLayer(block.conv1)
                block.conv2 = RescaleLayer(block.conv2)
    return net1.eval().to(device)


def reset_bn_stats(model, device, bn_loader, num_samples=None):
    # Reset batch norm statistics
    for m in model.modules():
        bn_calibration_init(m)
    model.train()
    with torch.no_grad():
        count = 0
        for data, _ in bn_loader:
            if num_samples is not None and count >= num_samples:
                break
            count += data.shape[0]
            data = data.to(device)
            model(data)
    model.eval()


class BatchScale2d(BatchNorm2d):
    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting
            #  this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization
          rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when
          buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (
                self.running_var is None)

        input_var = input.var([0, 2, 3])
        if bn_training and self.track_running_stats:
            self.running_var = (1 - exponential_average_factor) * self.running_var + exponential_average_factor * input_var
            return input / torch.sqrt(input_var[None, :, None, None] + self.eps) * self.weight[None, :, None, None]
        else:
            return input / torch.sqrt(self.running_var[None, :, None, None] + self.eps) * self.weight[None, :, None, None]


def repair(loader, model_tracked_s, model_repaired, device, alpha_s=None,
           variant='repair', average=False, factor=1, name='vgg', num_samples=None):
    model_tracked_s = [make_tracked_net(model, device, name) for model in model_tracked_s]
    for model in model_tracked_s:
        reset_bn_stats(model, device, loader, num_samples)

    if variant == 'repair':
        model_repaired = make_repaired_net(model_repaired, device, name=name)
    elif variant == 'rescale':
        model_repaired = make_rescale_net(model_repaired, device, name=name)

    num = len(model_tracked_s)
    if alpha_s is None:
        alpha_s = [1/num] * num

    for layers in zip(*[model_tracked.modules() for model_tracked in model_tracked_s],
                      model_repaired.modules()):

        if not isinstance(layers[0], TrackLayer):
            continue

        # get neuronal statistics of original networks
        mu_s = [layer.get_stats()[0] for layer in layers[:-1]]
        std_s = [layer.get_stats()[1] for layer in layers[:-1]]
        # set the goal neuronal statistics for the merged network
        goal_mean = sum([alpha * mu for alpha, mu in zip(alpha_s, mu_s)])
        goal_std = sum([alpha * std for alpha, std in zip(alpha_s, std_s)])
        if average:
            goal_mean = torch.ones_like(goal_mean) * goal_mean.abs().mean() * goal_mean.sign()
            goal_std = torch.ones_like(goal_std) * goal_std.mean()
        if variant == 'repair':
            layers[-1].set_stats(goal_mean * factor, goal_std * factor)
        elif variant == 'rescale':
            layers[-1].set_stats(goal_std * factor)

    reset_bn_stats(model_repaired, device, loader, num_samples)
    return model_repaired
