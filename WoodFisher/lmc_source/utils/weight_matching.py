from collections import defaultdict
from typing import NamedTuple
import torch
from scipy.optimize import linear_sum_assignment


class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict


def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes),
                           axes_to_perm=axes_to_perm)


def dense(name, p_in, p_out, bias=True):
    axes_to_perm = {f"{name}.weight": (p_out, p_in)}
    if bias is True:
        axes_to_perm.update({f"{name}.bias": (p_out, )})
    return axes_to_perm


def mlp_permutation_spec(num_hidden_layers: int, bias=True) -> PermutationSpec:
    """We assume that one permutation cannot appear in two axes of the same
    weight array."""
    assert num_hidden_layers >= 1
    if bias:
        bias_hidden = {f"layer{i}.bias": (f"P_{i}", )
                       for i in range(num_hidden_layers)}
        bias_last = {f"layer{num_hidden_layers}.bias": (None, )}
    else:
        bias_hidden, bias_last = {}, {}

    return permutation_spec_from_axes_to_perm({
        "layer0.weight": ("P_0", None),
        **{f"layer{i}.weight": (f"P_{i}", f"P_{i-1}")
           for i in range(1, num_hidden_layers)},
        **bias_hidden,
        f"layer{num_hidden_layers}.weight": (None, f"P_{num_hidden_layers-1}"),
        **bias_last,
    })


def lenet_permutation_spec() -> PermutationSpec:
    axes_to_perm = {
        "conv0.weight": ("P_Conv_0", None, None, None),
        "conv0.bias": ("P_Conv_0", ),  # "," is necessary
        "conv1.weight": ("P_Conv_1", "P_Conv_0", None, None),
        "conv1.bias": ("P_Conv_1", ),
        **dense("fc0", "P_Conv_1", "P_Dense_0"),
        **dense("fc1", "P_Dense_0", "P_Dense_1"),
        **dense("fc2", "P_Dense_1", None),
    }
    return permutation_spec_from_axes_to_perm(axes_to_perm)


def cnn_permuation_spec() -> PermutationSpec:
    axes_to_perm = {
        "0.weight": ("P_Conv_0", None, None, None),
        "2.weight": ("P_Conv_2", "P_Conv_0", None, None),
        "5.weight": ("P_Conv_5", "P_Conv_2", None, None),
        "7.weight": ("P_Conv_7", "P_Conv_5", None, None),
        **dense("11", "P_Conv_5", None),
    }
    return permutation_spec_from_axes_to_perm(axes_to_perm)

# Get permutation spec for VGG
# model = vgg.vgg11_bn(with_od=False)

# layers_with_conv = []
# layers_with_conv_b4 = []
# layers_with_bn = []

# for name, layer in model.named_modules():
    # if name.startswith('features') and isinstance(layer,
    #                                               (ODConv2d, nn.Conv2d)):
#         layers_with_conv.append(int(name.split('.')[-1]))
#     elif name.startswith('features') and isinstance(layer, BatchNorm2d):
#         layers_with_bn.append(int(name.split('.')[-1]))

# layers_with_conv = layers_with_conv[1:]
# layers_with_conv_b4 = [0] + deepcopy(layers_with_conv[:-1])
# layers_with_bn = layers_with_bn[1:]


vgg_layers_map = {'vgg11': [[3, 6, 8, 11, 13, 16, 18],
                            [0, 3, 6, 8, 11, 13, 16],
                            None],
                  'vgg13': [[2, 5, 7, 10, 12, 15, 17, 20, 22],
                            [0, 2, 5, 7, 10, 12, 15, 17, 20],
                            None],
                  'vgg16': [[2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28],
                            [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26],
                            None],
                  'vgg19': [[2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30,
                             32, 34],
                            [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28,
                             30, 32],
                            None],
                  'vgg11_bn': [[4, 8, 11, 15, 18, 22, 25],
                               [0, 4, 8, 11, 15, 18, 22],
                               [5, 9, 12, 16, 19, 23, 26]],
                  'vgg13_bn': [[3, 7, 10, 14, 17, 21, 24, 28, 31],
                               [0, 3, 7, 10, 14, 17, 21, 24, 28],
                               [4, 8, 11, 15, 18, 22, 25, 29, 32]],
                  'vgg16_bn': [[3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40],
                               [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37],
                               [4, 8, 11, 15, 18, 21, 25, 28, 31, 35, 38, 41]],
                  'vgg19_bn': [[3, 7, 10, 14, 17, 20, 23, 27, 30, 33, 36, 40,
                                43, 46, 49],
                               [0, 3, 7, 10, 14, 17, 20, 23, 27, 30, 33, 36,
                                40, 43, 46],
                               [4, 8, 11, 15, 18, 21, 24, 28, 31, 34, 37, 41,
                                44, 47, 50]],
                  'vgg11_ln': [[4, 8, 11, 15, 18, 22, 25],
                               [0, 4, 8, 11, 15, 18, 22],
                               [5, 9, 12, 16, 19, 23, 26]],
                  'vgg11_gn': [[4, 8, 11, 15, 18, 22, 25],
                               [0, 4, 8, 11, 15, 18, 22],
                               [5, 9, 12, 16, 19, 23, 26]],
                  'vgg13_ln': [[3, 7, 10, 14, 17, 21, 24, 28, 31],
                               [0, 3, 7, 10, 14, 17, 21, 24, 28],
                               [4, 8, 11, 15, 18, 22, 25, 29, 32]],
                  'vgg16_ln': [[3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40],
                               [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37],
                               [4, 8, 11, 15, 18, 21, 25, 28, 31, 35, 38, 41]],
                  'vgg19_ln': [[3, 7, 10, 14, 17, 20, 23, 27, 30, 33, 36, 40,
                                43, 46, 49],
                               [0, 3, 7, 10, 14, 17, 20, 23, 27, 30, 33, 36,
                                40, 43, 46],
                               [4, 8, 11, 15, 18, 21, 24, 28, 31, 34, 37, 41,
                                44, 47, 50]]}
additional = {}

additional['vgg16_nobias'] = vgg_layers_map['vgg16']
additional['vgg11_nobias'] = vgg_layers_map['vgg11']
vgg_layers_map.update(additional)


def vgg_permutation_spec(model: str) -> PermutationSpec:
    if model.endswith('x'):
        tmp = model.split('_')[:-1]
        model = '_'.join(tmp)
    elif model.endswith('nobias'):
        model = model[:-7]
    layers_with_conv = vgg_layers_map[model][0]
    layers_with_conv_b4 = vgg_layers_map[model][1]
    layers_with_norm = vgg_layers_map[model][2]

    if model.endswith('bn'):
        return permutation_spec_from_axes_to_perm({
            "features.0.weight": ("P_Conv_0", None, None, None),
            "features.1.weight": ("P_Conv_0", None),
            "features.1.bias": ("P_Conv_0", None),
            "features.1.running_mean": ("P_Conv_0", None),
            "features.1.running_var": ("P_Conv_0", None),
            "features.1.num_batches_tracked": (),

            **{f"features.{layers_with_conv[i]}.weight":
                (f"P_Conv_{layers_with_conv[i]}",
                 f"P_Conv_{layers_with_conv_b4[i]}", None, None, )
                for i in range(len(layers_with_conv))},
            **{f"features.{i}.bias": (f"P_Conv_{i}", )
               for i in layers_with_conv + [0]},
            # bn
            **{f"features.{layers_with_norm[i]}.weight":
                (f"P_Conv_{layers_with_conv[i]}", None)
                for i in range(len(layers_with_norm))},
            **{f"features.{layers_with_norm[i]}.bias":
                (f"P_Conv_{layers_with_conv[i]}", None)
                for i in range(len(layers_with_norm))},
            **{f"features.{layers_with_norm[i]}.running_mean":
                (f"P_Conv_{layers_with_conv[i]}", None)
                for i in range(len(layers_with_norm))},
            **{f"features.{layers_with_norm[i]}.running_var":
                (f"P_Conv_{layers_with_conv[i]}", None)
                for i in range(len(layers_with_norm))},
            **{f"features.{layers_with_norm[i]}.num_batches_tracked": ()
               for i in range(len(layers_with_norm))},

            **dense("classifier.0", f"P_Conv_{layers_with_conv[-1]}",
                    "P_Dense_0"),
            **dense("classifier.3", "P_Dense_0", "P_Dense_3"),
            **dense("classifier.6", "P_Dense_3", None)})
    elif model.endswith('ln'):
        return permutation_spec_from_axes_to_perm({
            "features.0.weight": ("P_Conv_0", None, None, None),
            "features.1.weight": ("P_Conv_0", None),
            "features.1.bias": ("P_Conv_0", None),

            **{f"features.{layers_with_conv[i]}.weight":
                (f"P_Conv_{layers_with_conv[i]}",
                 f"P_Conv_{layers_with_conv_b4[i]}", None, None, )
                for i in range(len(layers_with_conv))},
            **{f"features.{i}.bias": (f"P_Conv_{i}", )
               for i in layers_with_conv + [0]},
            # ln
            **{f"features.{layers_with_norm[i]}.weight":
                (f"P_Conv_{layers_with_conv[i]}", None)
                for i in range(len(layers_with_norm))},
            **{f"features.{layers_with_norm[i]}.bias":
                (f"P_Conv_{layers_with_conv[i]}", None)
                for i in range(len(layers_with_norm))},

            **dense("classifier.0", f"P_Conv_{layers_with_conv[-1]}",
                    "P_Dense_0"),
            **dense("classifier.3", "P_Dense_0", "P_Dense_3"),
            **dense("classifier.6", "P_Dense_3", None)})
    elif model.endswith('gn'):
        return permutation_spec_from_axes_to_perm({
            "features.0.weight": ("P_Conv_0", None, None, None),
            "features.1.weight": ("P_Conv_0", None),
            "features.1.bias": ("P_Conv_0", None),

            **{f"features.{layers_with_conv[i]}.weight":
                (f"P_Conv_{layers_with_conv[i]}",
                 f"P_Conv_{layers_with_conv_b4[i]}", None, None, )
                for i in range(len(layers_with_conv))},
            **{f"features.{i}.bias": (f"P_Conv_{i}", )
               for i in layers_with_conv + [0]},
            # gn
            **{f"features.{layers_with_norm[i]}.weight":
                (f"P_Conv_{layers_with_conv[i]}", None)
                for i in range(len(layers_with_norm))},
            **{f"features.{layers_with_norm[i]}.bias":
                (f"P_Conv_{layers_with_conv[i]}", None)
                for i in range(len(layers_with_norm))},

            **dense("classifier.0", f"P_Conv_{layers_with_conv[-1]}",
                    "P_Dense_0"),
            **dense("classifier.3", "P_Dense_0", "P_Dense_3"),
            **dense("classifier.6", "P_Dense_3", None)})
    else:
        return permutation_spec_from_axes_to_perm({
            # first features
            "features.0.weight": ("P_Conv_0", None, None, None),

            **{f"features.{layers_with_conv[i]}.weight":
                (f"P_Conv_{layers_with_conv[i]}",
                 f"P_Conv_{layers_with_conv_b4[i]}", None, None, )
                for i in range(len(layers_with_conv))},
            **{f"features.{i}.bias": (f"P_Conv_{i}", )
                for i in layers_with_conv + [0]},

            **dense("classifier.0", f"P_Conv_{layers_with_conv[-1]}",
                    "P_Dense_0"),
            **dense("classifier.3", "P_Dense_0", "P_Dense_3"),
            **dense("classifier.6", "P_Dense_3", None),
            })


def cifar_vgg_permutation_spec(model: str) -> PermutationSpec:
    if model.endswith('x'):
        tmp = model.split('_')[:-1]
        model = '_'.join(tmp)
    layers_with_conv = vgg_layers_map[model[6:]][0]
    layers_with_conv_b4 = vgg_layers_map[model[6:]][1]
    layers_with_norm = vgg_layers_map[model[6:]][2]

    if model.endswith('bn'):
        return permutation_spec_from_axes_to_perm({
            "features.0.weight": ("P_Conv_0", None, None, None),
            "features.1.weight": ("P_Conv_0", None),
            "features.1.bias": ("P_Conv_0", None),
            "features.1.running_mean": ("P_Conv_0", None),
            "features.1.running_var": ("P_Conv_0", None),
            "features.1.num_batches_tracked": (),

            **{f"features.{layers_with_conv[i]}.weight":
                (f"P_Conv_{layers_with_conv[i]}",
                 f"P_Conv_{layers_with_conv_b4[i]}", None, None, )
                for i in range(len(layers_with_conv))},
            **{f"features.{i}.bias": (f"P_Conv_{i}", )
               for i in layers_with_conv + [0]},
            # bn
            **{f"features.{layers_with_norm[i]}.weight":
                (f"P_Conv_{layers_with_conv[i]}", None)
                for i in range(len(layers_with_norm))},
            **{f"features.{layers_with_norm[i]}.bias":
                (f"P_Conv_{layers_with_conv[i]}", None)
                for i in range(len(layers_with_norm))},
            **{f"features.{layers_with_norm[i]}.running_mean":
                (f"P_Conv_{layers_with_conv[i]}", None)
                for i in range(len(layers_with_norm))},
            **{f"features.{layers_with_norm[i]}.running_var":
                (f"P_Conv_{layers_with_conv[i]}", None)
                for i in range(len(layers_with_norm))},
            **{f"features.{layers_with_norm[i]}.num_batches_tracked": ()
               for i in range(len(layers_with_norm))},

            **dense("classifier.0", f"P_Conv_{layers_with_conv[-1]}",
                    None)})
    elif model.endswith('ln'):
        return permutation_spec_from_axes_to_perm({
            "features.0.weight": ("P_Conv_0", None, None, None),
            "features.1.weight": ("P_Conv_0", None),
            "features.1.bias": ("P_Conv_0", None),

            **{f"features.{layers_with_conv[i]}.weight":
                (f"P_Conv_{layers_with_conv[i]}",
                 f"P_Conv_{layers_with_conv_b4[i]}", None, None, )
                for i in range(len(layers_with_conv))},
            **{f"features.{i}.bias": (f"P_Conv_{i}", )
               for i in layers_with_conv + [0]},
            # ln
            **{f"features.{layers_with_norm[i]}.weight":
                (f"P_Conv_{layers_with_conv[i]}", None)
                for i in range(len(layers_with_norm))},
            **{f"features.{layers_with_norm[i]}.bias":
                (f"P_Conv_{layers_with_conv[i]}", None)
                for i in range(len(layers_with_norm))},

            **dense("classifier.0", f"P_Conv_{layers_with_conv[-1]}",
                    None)})
    elif model.endswith('gn'):
        return permutation_spec_from_axes_to_perm({
            "features.0.weight": ("P_Conv_0", None, None, None),
            "features.1.weight": ("P_Conv_0", None),
            "features.1.bias": ("P_Conv_0", None),

            **{f"features.{layers_with_conv[i]}.weight":
                (f"P_Conv_{layers_with_conv[i]}",
                 f"P_Conv_{layers_with_conv_b4[i]}", None, None, )
                for i in range(len(layers_with_conv))},
            **{f"features.{i}.bias": (f"P_Conv_{i}", )
               for i in layers_with_conv + [0]},
            # gn
            **{f"features.{layers_with_norm[i]}.weight":
                (f"P_Conv_{layers_with_conv[i]}", None)
                for i in range(len(layers_with_norm))},
            **{f"features.{layers_with_norm[i]}.bias":
                (f"P_Conv_{layers_with_conv[i]}", None)
                for i in range(len(layers_with_norm))},

            **dense("classifier.0", f"P_Conv_{layers_with_conv[-1]}",
                    None)})
    elif 'nobias' in model:
        return permutation_spec_from_axes_to_perm({
            # first features
            "features.0.weight": ("P_Conv_0", None, None, None),

            **{f"features.{layers_with_conv[i]}.weight":
                (f"P_Conv_{layers_with_conv[i]}",
                 f"P_Conv_{layers_with_conv_b4[i]}", None, None, )
                for i in range(len(layers_with_conv))},

            **dense("classifier.0", f"P_Conv_{layers_with_conv[-1]}",
                    None)
            })
    else:
        return permutation_spec_from_axes_to_perm({
            # first features
            "features.0.weight": ("P_Conv_0", None, None, None),

            **{f"features.{layers_with_conv[i]}.weight":
                (f"P_Conv_{layers_with_conv[i]}",
                 f"P_Conv_{layers_with_conv_b4[i]}", None, None, )
                for i in range(len(layers_with_conv))},
            **{f"features.{i}.bias": (f"P_Conv_{i}", )
                for i in layers_with_conv + [0]},

            **dense("classifier.0", f"P_Conv_{layers_with_conv[-1]}",
                    None)
            })


def cifar_resnet_permutation_spec(model: str) -> PermutationSpec:
    if 'plain' in model or 'fixup' in model:
        norm = None
    elif 'ln' in model:
        norm = 'ln'
    else:
        norm = 'bn'

    bias = False
    if 'plain' in model and 'nobias' not in model:
        bias = True

    str_split = model.split('_')
    D = [0]
    for s in str_split:
        if 'resnet' in s:
            D[0] = (int(s[6:]) - 2) // 6
            break

    if bias:
        conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None,
                                                            None, ),
                                          f"{name}.bias": (p_out, )}
    else:
        conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None,
                                                            None, )}
    conv_nobias = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in,
                                                                 None, None, )}

    if norm == 'bn':
        norm = lambda name, p: {f"{name}.weight": (p, ), f"{name}.bias": (p, ),
                                f"{name}.running_mean": (p, ),
                                f"{name}.running_var": (p, ),
                                f"{name}.num_batches_tracked": ()}
    elif norm == 'ln':
        norm = lambda name, p: {f"{name}.weight": (p, ), f"{name}.bias": (p, )}
    elif norm is None:
        norm = lambda name, p: {}

    dense = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in),
                                       f"{name}.bias": (p_out, )}
    if 'nobias' in model:
        dense = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in)}

    # This is for easy blocks that use a residual connection, without any
    # change in the number of channels.
    easyblock = lambda name, p: {
        **conv(f"{name}.conv1", p, f"P_{name}_inner"),
        **norm(f"{name}.norm1", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p),
        **norm(f"{name}.norm2", p)
    }

    # This is for blocks that use a residual connection, but change the number
    # of channels via a Conv.
    shortcutblock = lambda name, p_in, p_out: {
        **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
        **norm(f"{name}.norm1", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
        **norm(f"{name}.norm2", p_out),
        **conv_nobias(f"{name}.shortcut.0", p_in, p_out),
        **norm(f"{name}.shortcut.1", p_out),
        }

    axes_to_perm = {**conv("conv1", None, "P_bg0"),
                    **norm("norm1", "P_bg0")}
    for i in range(D[0]):
        axes_to_perm.update(**easyblock(f"segments.0.{i}", "P_bg0"))
    axes_to_perm.update(**shortcutblock("segments.1.0", "P_bg0", "P_bg1"))
    for i in range(1, D[0]):
        axes_to_perm.update(**easyblock(f"segments.1.{i}", "P_bg1"))
    axes_to_perm.update(**shortcutblock("segments.2.0", "P_bg1", "P_bg2"))
    for i in range(1, D[0]):
        axes_to_perm.update(**easyblock(f"segments.2.{i}", "P_bg2"))
    axes_to_perm.update(**dense("fc", "P_bg2", None))

    return permutation_spec_from_axes_to_perm(axes_to_perm)


def get_permuted_param(ps: PermutationSpec, perm, k: str, params,
                       except_axis=None, device=None):
    """Get parameter `k` from `params`, with the permutations applied."""
    w = params[k]
    if k in ps.axes_to_perm:
        for axis, p in enumerate(ps.axes_to_perm[k]):
            # Skip the axis we're trying to permute.
            if axis == except_axis:
                continue

            # None indicates that there is no permutation relevant to that axis.
            if p is not None:
                if len(perm[p]) == w.shape[axis]:
                    w = torch.index_select(w, axis, perm[p].int())
                else:
                    # this doesn't happen in current settings
                    perm_group = perm[p].int()
                    perm_ori = []
                    size = int(w.shape[axis] / len(perm_group))
                    for id in perm_group:
                        perm_ori.extend(range(id*size, (id+1)*size))
                    w = torch.index_select(w, axis,
                                        torch.tensor(perm_ori, device=device))

    return w


def apply_permutation(ps: PermutationSpec, perm, params, device=None):
    """Apply a `perm` to `params`."""
    return {k: get_permuted_param(ps, perm, k, params, device=device)
            for k in params.keys()}


def weight_matching(model_name, params_a, params_b, max_iter=100,
                    init_perm=None, device=None, return_perm=False,
                    tol=1e-12):
    """return model_b's parameters after weight matching"""
    ps = get_permutation_spec(model_name)
    perm = get_wm_perm(ps, params_a, params_b, max_iter=max_iter,
                       init_perm=init_perm, device=device, tol=tol)
    params_b_perm = apply_permutation(ps, perm, params_b, device=device)
    if return_perm:
        return params_b_perm, perm
    else:
        return params_b_perm, None


def get_wm_perm(ps, params_a, params_b, max_iter=100,
                init_perm=None, device=None, tol=1e-12):
    """Find a permutation of `params_b` to make them match `params_a`."""
    perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]]
                  for p, axes in ps.perm_to_axes.items()}

    perm = {p: torch.arange(n, device=device) for p, n in perm_sizes.items()
            } if init_perm is None else init_perm
    perm_names = list(perm.keys())

    for iteration in range(max_iter):
        progress = False
        # for p_ix in range(len(perm_names)):
        for p_ix in torch.randperm(len(perm_names)):

            p = perm_names[p_ix]
            n = perm_sizes[p]
            A = torch.zeros((n, n), device=device)
            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a[wk]
                w_b = get_permuted_param(
                    ps, perm, wk, params_b, except_axis=axis, device=device)
                w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1))
                w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1))

                A += w_a @ w_b.T

            ri, ci = linear_sum_assignment(A.cpu().detach().numpy(),
                                           maximize=True)
            assert (torch.tensor(ri) == torch.arange(len(ri))).all()
            oldL = torch.einsum('ij,ij->i', A, torch.eye(n, device=device)
                                [perm[p].long()]).sum()
            newL = torch.einsum('ij,ij->i', A,
                                torch.eye(n, device=device)[ci, :]).sum()
            print(f"{iteration}/{p}:")
            print(f"oldL: {oldL}")
            print(f"newL: {newL}")
            print(f"newL - oldL: {newL - oldL}")
            progress = progress or newL > oldL + tol

            perm[p] = torch.Tensor(ci).to(device)

        if not progress:
            break

    return perm


def get_permutation_spec(model):
    if model == 'simple_mlp':
        ps = mlp_permutation_spec(num_hidden_layers=1, bias=False)
    elif model == 'lenet':
        ps = lenet_permutation_spec()
    elif model.startswith('vgg'):
        ps = vgg_permutation_spec(model)
    elif model.startswith('cifar_vgg'):
        ps = cifar_vgg_permutation_spec(model)
    elif 'cifar_resnet' in model:
        ps = cifar_resnet_permutation_spec(model)
    elif model == 'mlp':
        ps = mlp_permutation_spec(num_hidden_layers=2)
    elif model == 'mnist_mlp':
        ps = mlp_permutation_spec(num_hidden_layers=3)
    elif model == 'mlp_mnist':
        ps = mlp_permutation_spec(num_hidden_layers=3)
    elif model == 'cnn':
        ps = cnn_permuation_spec()
    else:
        raise ValueError(f"Unknown model {model}")
    return ps
