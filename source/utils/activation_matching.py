import torch.nn as nn
import torch
from einops import rearrange
from scipy.optimize import linear_sum_assignment

# from source.od.models.vgg import vgg16
import source.utils.weight_matching as wm
from source.utils.online_stats import OnlineMean, OnlineCovariance, OnlineCovariance_Git


def activation_matching(model_name, model_a, model_b, loader, print_freq,
                        device=None, return_perm=False, pre_relu=False,
                        type=None):
    """return model_b's parameters after activation matching"""
    ps = wm.get_permutation_spec(model_name)
    if type is None:
        perm = get_am_perm(model_name, model_a, model_b, loader,
                        print_freq=print_freq, device=device,
                        pre_relu=pre_relu)
    elif type == 'git':
        perm = get_am_perm_git(model_name, model_a, model_b, loader,
                        print_freq=print_freq, device=device,
                        pre_relu=pre_relu)
    else:
        raise NotImplementedError(f'type {type} not implemented')

    sd_b_perm = apply_permutation(ps, perm, model_b.state_dict(), device)
    if return_perm:
        return sd_b_perm, perm
    else:
        return sd_b_perm, None


def get_am_perm(model_name, model_a, model_b, loader, print_freq,
                device=None, pre_relu=False):
    """return a permutation to match model_b's parameters to model_a's."""
    # get intermediate layers
    intermediate_layers = get_intermediate_layers(model_name, model_a)
    # record intermediate layers
    model_a.record(intermediate_layers, pre_act=pre_relu)
    model_b.record(intermediate_layers, pre_act=pre_relu)
    # set model to eval mode
    model_a.eval()
    model_b.eval()
    # get feature dimension use a dummy input
    dummy_input = loader.dataset[0][0].unsqueeze(0).to(device)
    with torch.no_grad():
        model_a(dummy_input)
    acts_a = model_a.selected_out
    feature_nums = {layer_name: acts_a[layer_name].shape[1]
                    for layer_name in intermediate_layers}
    # calculate mean
    means_a = {layer_name: OnlineMean(feature_nums[layer_name], device) for
               layer_name in intermediate_layers}
    means_b = {layer_name: OnlineMean(feature_nums[layer_name], device) for
               layer_name in intermediate_layers}
    # run one epoch
    num_batches = len(loader)
    covs = {layer_name: OnlineCovariance(means_a[layer_name].mean(),
                                         means_b[layer_name].mean(),
                                         num_batches, device)
            for layer_name in intermediate_layers}
    for data, _ in loader:
        with torch.no_grad():
            data = data.to(device)
            # forward
            model_a(data)
            model_b(data)
            # get activation
            acts_a = model_a.selected_out
            acts_b = model_b.selected_out
            # update covariance for each layerS
            for layer_name in intermediate_layers:
                act_a = acts_a[layer_name]
                act_b = acts_b[layer_name]
                # flatten activations: 'b c w h -> c (b h w)'
                # or 'b c -> c b'
                if len(act_a.shape) == 4:
                    act_a = rearrange(act_a, 'b c w h -> c (b h w)')
                    act_b = rearrange(act_b, 'b c w h -> c (b h w)')
                elif len(act_a.shape) == 2:
                    act_a = rearrange(act_a, 'b c -> c b')
                    act_b = rearrange(act_b, 'b c -> c b')
                covs[layer_name].update(act_a, act_b)
    perm_values = []
    # calculate permutation
    for layer_name in intermediate_layers:
        correlation = covs[layer_name].pearson_correlation()
        ri, ci = linear_sum_assignment(correlation.cpu().detach().numpy(),
                                       maximize=True)
        ci = torch.from_numpy(ci).to(device)
        perm_values.append(ci)
        oldL = torch.einsum('ij,ij->i', correlation,
                            torch.eye(len(ci), device=device, dtype=torch.double)).sum()
        newL = torch.einsum('ij,ij->i', correlation,
                            torch.eye(len(ci), device=device, dtype=torch.double)[ci, :]).sum()
        print(f"0/{layer_name}")
        print(f"oldL: {oldL}")
        print(f"newL: {newL}")
        print(f"newL - oldL: {newL - oldL}")
    # store permutation
    ps = wm.get_permutation_spec(model_name)
    perm_keys = list(ps.perm_to_axes.keys())
    assert len(perm_keys) == len(perm_values)
    perm = {k: v for k, v in zip(perm_keys, perm_values)}
    # compare l

    # stop recording
    model_a.stop_record()
    model_b.stop_record()

    return perm


def get_am_perm_git(model_name, model_a, model_b, loader, print_freq,
                    device=None, pre_relu=False):
    """return a permutation to match model_b's parameters to model_a's."""
    # get intermediate layers
    intermediate_layers = get_intermediate_layers(model_name, model_a)
    # record intermediate layers
    model_a.record(intermediate_layers, pre_act=pre_relu)
    model_b.record(intermediate_layers, pre_act=pre_relu)
    # set model to eval mode
    model_a.eval()
    model_b.eval()
    # get feature dimension use a dummy input
    dummy_input = loader.dataset[0][0].unsqueeze(0).to(device)
    with torch.no_grad():
        model_a(dummy_input)
    acts_a = model_a.selected_out
    feature_nums = {layer_name: acts_a[layer_name].shape[1]
                    for layer_name in intermediate_layers}
    # calculate mean
    means_a = {layer_name: OnlineMean(feature_nums[layer_name], device) for
               layer_name in intermediate_layers}
    means_b = {layer_name: OnlineMean(feature_nums[layer_name], device) for
               layer_name in intermediate_layers}
    # run one epoch
    num_batches = len(loader)
    id_batch = 0
    for data, _ in loader:
        if id_batch % print_freq == 0 or id_batch == num_batches - 1:
            print('batch {}/{}'.format(id_batch+1, num_batches))
        with torch.no_grad():
            data = data.to(device)
            # forward
            model_a(data)
            model_b(data)
            # get activation
            acts_a = model_a.selected_out
            acts_b = model_b.selected_out
            # update mean for each layer
            for layer_name in intermediate_layers:
                act_a = acts_a[layer_name]
                act_b = acts_b[layer_name]
                means_a[layer_name].update(act_a)
                means_b[layer_name].update(act_b)
        id_batch += 1
    # calculate covariance
    covs = {layer_name: OnlineCovariance_Git(means_a[layer_name].mean(),
                                         means_b[layer_name].mean(),
                                         device)
            for layer_name in intermediate_layers}
    # run one epoch
    for data, _ in loader:
        with torch.no_grad():
            data = data.to(device)
            # forward
            model_a(data)
            model_b(data)
            # get activation
            acts_a = model_a.selected_out
            acts_b = model_b.selected_out
            # update covariance for each layerS
            for layer_name in intermediate_layers:
                act_a = acts_a[layer_name]
                act_b = acts_b[layer_name]
                # flatten activations: 'b c w h -> c (b h w)'
                # or 'b c -> c b'
                if len(act_a.shape) == 4:
                    act_a = rearrange(act_a, 'b c w h -> c (b h w)')
                    act_b = rearrange(act_b, 'b c w h -> c (b h w)')
                elif len(act_a.shape) == 2:
                    act_a = rearrange(act_a, 'b c -> c b')
                    act_b = rearrange(act_b, 'b c -> c b')
                covs[layer_name].update(act_a, act_b)
    perm_values = []
    # calculate permutation
    for layer_name in intermediate_layers:
        correlation = covs[layer_name].pearson_correlation()
        ri, ci = linear_sum_assignment(correlation.cpu().detach().numpy(),
                                       maximize=True)
        ci = torch.from_numpy(ci).to(device)
        perm_values.append(ci)
        oldL = torch.einsum('ij,ij->i', correlation,
                            torch.eye(len(ci), device=device)).sum()
        newL = torch.einsum('ij,ij->i', correlation,
                            torch.eye(len(ci), device=device)[ci, :]).sum()
        print(f"0/{layer_name}")
        print(f"oldL: {oldL}")
        print(f"newL: {newL}")
        print(f"newL - oldL: {newL - oldL}")
    # store permutation
    ps = wm.get_permutation_spec(model_name)
    perm_keys = list(ps.perm_to_axes.keys())
    assert len(perm_keys) == len(perm_values)
    perm = {k: v for k, v in zip(perm_keys, perm_values)}
    # compare l

    # stop recording
    model_a.stop_record()
    model_b.stop_record()

    return perm


def get_intermediate_layers(model_name: str, model):
    intermediate_layers = []
    if model_name.startswith('vgg') or model_name.startswith('cifar_vgg'):
        # didn' use (consider) dropout
        for layer_name, layer in model.named_modules():
            if isinstance(layer, nn.ReLU):
                intermediate_layers.append(layer_name)
    elif 'cifar_resnet' in model_name:
        str_split = model_name.split('_')
        D = [0]
        for s in str_split:
            if 'resnet' in s:
                D[0] = (int(s[6:]) - 2) // 6
                break
        intermediate_layers = [f'segments.0.{D[0]-1}.relu2']
        intermediate_layers.extend([f'segments.0.{i}.relu1' for i in range(D[0])])
        intermediate_layers.append('segments.1.0.relu1')
        intermediate_layers.append(f'segments.1.{D[0]-1}.relu2')
        intermediate_layers.extend([f'segments.1.{i}.relu1' for i in range(1, D[0])])
        intermediate_layers.append('segments.2.0.relu1')
        intermediate_layers.append(f'segments.2.{D[0]-1}.relu2')
        intermediate_layers.extend([f'segments.2.{i}.relu1' for i in range(1, D[0])])

    elif model_name == 'simple_mlp':
        # should use relu
        for layer_name, layer in model.named_modules():
            if isinstance(layer, nn.ReLU):
                intermediate_layers.append(layer_name)
    elif model_name in ['mlp', 'mlp_mnist', 'lenet']:
        for layer_name, layer in model.named_modules():
            if isinstance(layer, nn.ReLU):
                intermediate_layers.append(layer_name)
    else:
        raise NotImplementedError(f'model {model_name} not implemented')
    return intermediate_layers


def apply_permutation(ps, perm, params, device=None):
    return wm.apply_permutation(ps, perm, params, device)
