import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from lmc_source.utils.train import validate
from lmc_source.od.layers import ODConv2d
from lmc_source.od.layers.batch_norm import bn_calibration_init
from torch import Tensor
from torch.nn import BatchNorm2d
# from torch.nn import functional as F
from utils.masking_utils import WrappedLayer


def get_weights(model):
    """Get weights of a model as a numpy array"""
    if isinstance(model, torch.nn.Module):
        # model; exclude bn statistics
        return np.concatenate([p.data.cpu().numpy().ravel()
                               for p in model.parameters()])

    elif isinstance(model, dict):
        # state dict; include bn statistics
        weights = []
        for name, p in model.items():
            if 'num_batches_tracked' in name:
                continue
            else:
                weights.append(p.data.cpu().numpy().ravel())
        return np.concatenate(weights)


def interpolate_state_dicts(state_dict_1, state_dict_2, weight,
                            bias_norm=False):
    if not bias_norm:
        return {key: (1 - weight) * state_dict_1[key] +
                weight * state_dict_2[key] for key in state_dict_1.keys()}
    else:
        model_state = deepcopy(state_dict_1)
        height = 0
        for p_name in model_state:
            if "batches" not in p_name:
                model_state[p_name].zero_()
                if "weight" in p_name:
                    model_state[p_name].add_(1.0 - weight, state_dict_1[p_name])
                    model_state[p_name].add_(weight, state_dict_2[p_name])
                    height += 1
                if "bias" in p_name:
                    model_state[p_name].add_((1.0 - weight)**height, state_dict_1[p_name])
                    model_state[p_name].add_(weight**height, state_dict_2[p_name])
                if "res_scale" in p_name:
                    model_state[p_name].add_(1.0 - weight, state_dict_1[p_name])
                    model_state[p_name].add_(weight, state_dict_2[p_name])
        return model_state


def interpolate_multi_state_dicts(sd_s, weight_s):
    sd_interpolated = deepcopy(sd_s[0])
    for key in sd_s[0].keys():
        sd_interpolated[key] = weight_s[0] * sd_s[0][key]
        for i in range(1, len(sd_s)):
            sd_interpolated[key] += weight_s[i] * sd_s[i][key]
    return sd_interpolated


def calculate_models_distance(model_1, model_2):
    # TODO: improve choices of distance metrics
    w_1 = get_weights(model_1)
    w_2 = get_weights(model_2)
    distance = np.linalg.norm(w_1 - w_2)
    return distance


def eval_line(model_1, model_2, val_loader, criterion, device, config,
              reset_bn=False, bn_loader=None, n=11, repair=None, name='vgg',
              bias_norm=False):
    """Evaluate a line segment between two models"""
    alphas = np.linspace(0.0, 1.0, n)
    sd_1 = model_1.state_dict()
    sd_2 = model_2.state_dict()

    lmc_stat = [None] * n
    # test end points
    lmc_stat[0] = validate(val_loader, model_1,
                           criterion, device, config)
    lmc_stat[-1] = validate(val_loader, model_2,
                            criterion, device, config)
    if repair is not None:
        wrap_1 = make_tracked_net(model_1, device, name=name)
        wrap_2 = make_tracked_net(model_2, device, name=name)
        reset_bn_stats(wrap_1, device, bn_loader)
        reset_bn_stats(wrap_2, device, bn_loader)

    for i, alpha in enumerate(alphas[1:-1], 1):
        base_model = deepcopy(model_1)
        base_model.load_state_dict(interpolate_state_dicts(sd_1, sd_2, alpha,
                                                           bias_norm=bias_norm))
        if reset_bn:
            reset_bn_stats(base_model, device, bn_loader)
        if repair is not None:
            if repair == 'repair':
                wrap_a = make_repaired_net(base_model, device, name=name)
            elif repair == 'rescale':
                wrap_a = make_rescale_net(base_model, device, name=name)
            elif repair == 'reshift':
                wrap_a = make_reshift_net(base_model, device, name=name)

            for track_1, track_2, reset_a in zip(wrap_1.modules(),
                                                 wrap_2.modules(),
                                                 wrap_a.modules()):
                if not isinstance(track_1, TrackLayer):
                    continue
                assert (isinstance(track_1, TrackLayer)
                        and isinstance(track_2, TrackLayer))

                # get neuronal statistics of original networks
                mu_1, std_1 = track_1.get_stats()
                mu_2, std_2 = track_2.get_stats()
                # set the goal neuronal statistics for the merged network
                goal_mean = (1 - alpha) * mu_1 + alpha * mu_2
                goal_std = (1 - alpha) * std_1 + alpha * std_2
                if repair == 'repair':
                    reset_a.set_stats(goal_mean, goal_std)
                elif repair == 'rescale':
                    reset_a.set_stats(goal_std)
                elif repair == 'reshift':
                    reset_a.set_stats(goal_mean)
            reset_bn_stats(wrap_a, device, bn_loader)
            base_model = wrap_a

        lmc_stat[i] = validate(val_loader, base_model, criterion,
                               device, config)
    lmc_stat = torch.tensor(lmc_stat)
    return lmc_stat


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


# adds ResetLayers around every conv layer
def make_reshift_net(net, device=None, name='vgg'):
    net1 = deepcopy(net).to(device)
    if 'vgg' in name:
        for i, layer in enumerate(net1.features):
            if isinstance(layer, (nn.Conv2d)):
                net1.features[i] = ReshiftLayer(layer)
    elif 'resnet20' in name:
        for block_group in net1.segments:
            for block in block_group:
                block.conv1 = ReshiftLayer(block.conv1)
                block.conv2 = ReshiftLayer(block.conv2)
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


def fuse_conv_bn(conv, bn, device=None):
    fused_conv = deepcopy(conv).to(device)

    # set weights
    w_conv = conv.weight.clone()
    bn_std = (bn.eps + bn.running_var).sqrt()
    gamma = bn.weight / bn_std
    fused_conv.weight.data = (w_conv * gamma.reshape(-1, 1, 1, 1))

    # set bias
    beta = bn.bias + gamma * (-bn.running_mean + conv.bias)
    fused_conv.bias.data = beta

    return fused_conv


def fuse_tracked_net(net, device):
    net1 = deepcopy(net)
    for i, rlayer in enumerate(net.features):
        if isinstance(rlayer, (ResetLayer, RescaleLayer, ReshiftLayer)):
            fused_conv = fuse_conv_bn(rlayer.layer, rlayer.bn)
            net1.features[i] = fused_conv
    return net1.to(device)


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


class BatchShift2d(BatchNorm2d):
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

        input_mean = input.mean([0, 2, 3])
        # input_var = input.var([0, 2, 3])
        if bn_training and self.track_running_stats:
            self.running_mean = (1 - exponential_average_factor) * self.running_mean + exponential_average_factor * input_mean
            # self.running_var = (1 - exponential_average_factor) * self.running_var + exponential_average_factor * input_var
            return (input - input_mean[None, :, None, None]) + self.bias[None, :, None, None]
            # return (input - input_mean[None, :, None, None]) / torch.sqrt(input_var[None, :, None, None] + self.eps) * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        else:
            return (input - self.running_mean[None, :, None, None]) + self.bias[None, :, None, None]
            # return (input - self.running_mean[None, :, None, None]) / torch.sqrt(self.running_var[None, :, None, None] + self.eps) * self.weight[None, :, None, None] + self.bias[None, :, None, None]


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

class ReshiftLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.bn = BatchShift2d(layer._layer.out_channels)

    def set_stats(self, goal_mean):
        self.bn.bias.data = goal_mean

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


def repair(loader, model_tracked_s, model_repaired, device, alpha_s=None,
           variant='repair', average=False, factor=1, name='vgg', num_samples=None):
    model_tracked_s = [make_tracked_net(model, device, name) for model in model_tracked_s]
    for model in model_tracked_s:
        reset_bn_stats(model, device, loader, num_samples)

    if variant == 'repair':
        model_repaired = make_repaired_net(model_repaired, device, name=name)
    elif variant == 'rescale':
        model_repaired = make_rescale_net(model_repaired, device, name=name)
    elif variant == 'reshift':
        model_repaired = make_reshift_net(model_repaired, device, name=name)

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
        elif variant == 'reshift':
            layers[-1].set_stats(goal_mean)


    reset_bn_stats(model_repaired, device, loader, num_samples)
    return model_repaired
