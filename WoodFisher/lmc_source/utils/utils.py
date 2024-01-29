import os
import torch
import shutil
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import random
from enum import Enum

from .logger import Logger
from ..od.models.lenet import lenet, lenet_t
from ..od.models.mlp import mlp, mnist_mlp, simple_mlp
from ..od.models import vgg, resnet, cifar_vgg, cifar_resnet, fixup_cifar_resnet, plain_cifar_resnet
from ..od.models.cnn import SimpleCNN
# from ..od.models.utils import create_linear_layer, create_conv2d_layer


def save_checkpoint(state, is_best, dir, filename='checkpoint.pth.tar'):
    model_dir = os.path.join(dir, filename)
    torch.save(state, model_dir)
    if is_best:
        shutil.copyfile(
            model_dir, os.path.join(dir, 'model_best.pth.tar')
        )


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        Logger.get().info('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        Logger.get().info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions
       for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def create_experiment_dir(args):
    run_id = f'id={args.identifier}'
    dir_name = f'{args.model}_{args.dataset}'

    if getattr(args, 'use_od_reg', None):
        reg_dir_name = 'od'
    else:
        reg_dir_name = 'plain'

    lr_dir_name = f'lr_{str(args.lr)}'
    seed_dir_name = f'seed_{str(args.seed)}'

    experiment_dir = os.path.join(
        args.outputs_dir, run_id, dir_name,
        reg_dir_name, lr_dir_name, seed_dir_name
    )
    return experiment_dir


def create_metrics_dict():
    return {
        'epoch': [],
        'loss': [],
        'top1': [],
        'top5': [],
        'time': [],
    }


def append_to_metrics_dict(metrics_dict, loss, epoch,
                           top1, top5, batch_time):
    metrics_dict['epoch'].append(epoch)
    metrics_dict['loss'].append(loss)
    metrics_dict['top1'].append(top1.item())
    metrics_dict['top5'].append(top5.item())
    metrics_dict['time'].append(batch_time)


def load_model(config, in_channels=3, return_func=False):
    if config.dataset == 'cifar10':
        num_classes = 10
    elif config.dataset == 'cifar100':
        num_classes = 100

    if config.model == 'lenet':
        model = lenet(in_channels=in_channels,
                      with_od=config.with_od,
                      special_init=config.special_init)
        if return_func:
            def model_func():
                return lenet(in_channels=in_channels,
                             with_od=config.with_od,
                             special_init=config.special_init)
    elif config.model.endswith('_toroids'):
        model_name = config.model[:-8]
        model_f = getattr(lenet_toroids, model_name)
        model = model_f(in_channels=in_channels)
    elif config.model == 'lenet_t':
        model = lenet_t(in_channels=in_channels,
                        with_od=config.with_od,
                        special_init=config.special_init)
        if return_func:
            def model_func():
                return lenet_t(in_channels=in_channels,
                               with_od=config.with_od,
                               special_init=config.special_init)

    elif config.model == 'mlp':
        model = mlp(in_channels=in_channels,
                    with_od=config.with_od)
        if return_func:
            def model_func():
                return mlp(in_channels=in_channels,
                           with_od=config.with_od)

    elif config.model == 'mnist_mlp':
        model = mnist_mlp(in_channels=in_channels,
                          with_od=config.with_od,
                          mnist_mlp_init_ratio=config.mnist_mlp_init_ratio)
        if return_func:
            def model_func():
                return mnist_mlp(in_channels=in_channels,
                                 with_od=config.with_od,
                                 mnist_mlp_init_ratio=config.mnist_mlp_init_ratio)

    elif config.model == 'simple_mlp':
        model = simple_mlp(with_od=config.with_od,
                           with_relu=config.with_relu,
                           special_init=config.special_init)
        if return_func:
            def model_func():
                return simple_mlp(with_od=config.with_od,
                                  with_relu=config.with_relu,
                                  special_init=config.special_init)
    elif config.model == 'cnn':
        model = SimpleCNN(config.dataset)

    elif config.model.startswith('vgg'):
        if config.model.endswith('_toroids'):
            model_name = config.model[:-8]
            model_f = getattr(vgg_toroids, model_name)
            model = model_f(num_classes=num_classes)
        else:
            model_f = getattr(vgg, config.model)
            model = model_f(num_classes=num_classes,
                            with_od=config.with_od,
                            with_dp=config.with_dp,
                            special_init=config.special_init,
                            git_rebasin_model=config.git_rebasin_model)
            if return_func:
                def model_func():
                    return model_f(num_classes=num_classes,
                                   with_od=config.with_od,
                                   with_dp=config.with_dp,
                                   special_init=config.special_init,
                                   git_rebasin_model=config.git_rebasin_model)

    elif config.model.startswith('cifar_vgg'):
        last = config.model.split('_')[-1]

        if last.endswith('x'):
            width_multiplier = int(last[:-1])
            model_f = getattr(cifar_vgg, config.model[:-len(last)-1])
        else:
            width_multiplier = 1
            model_f = getattr(cifar_vgg, config.model)
        model = model_f(num_classes=num_classes,
                        with_od=config.with_od,
                        special_init=config.special_init,
                        width_multiplier=width_multiplier)
        if return_func:
            def model_func():
                return model_f(num_classes=num_classes,
                               with_od=config.with_od,
                               special_init=config.special_init,
                               width_multiplier=width_multiplier)

    elif config.model.startswith('cifar_resnet'):
        model_name = config.model
        model = cifar_resnet.ResNet.get_model_from_name(model_name,
                                                        num_classes,
                                                        config.with_od,
                                                        config.special_init)
    elif config.model.startswith('fixup_cifar_resnet'):
        model_name = config.model
        model = fixup_cifar_resnet.ResNet.get_model_from_name(model_name,
                                                              num_classes,
                                                              config.with_od,
                                                              config.special_init)
    elif config.model.startswith('plain_cifar_resnet'):
        model_name = config.model
        model = plain_cifar_resnet.ResNet.get_model_from_name(model_name,
                                                              num_classes,
                                                              config.with_od,
                                                              config.special_init)
    else:
        raise ValueError('invalid model %r' % config.model)

    return model if not return_func else model_func


def load_optimizer(parameters, config):
    if config.optimizer == 'sgd':
        optimizer = optim.SGD(parameters, lr=config.lr,
                              momentum=config.momentum,
                              weight_decay=config.wd,
                              nesterov=config.nesterov)
    elif config.optimizer == 'adam':
        optimizer = optim.Adam(parameters, lr=config.lr,
                               weight_decay=config.wd)
    else:
        raise ValueError('invalid optimizer %r' % config.optimizer)
    return optimizer


def load_scheduler(optimizer, config):
    if config.scheduler == 'lambda':
        # Drop lr by 10 at each milestone
        milestones = [int(x) for x in config.milestones.split(',')]

        def lr_lambda(it):
            return 0.1 ** (sum(it >= np.array(milestones)))
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif config.scheduler == 'linear_cosine':
        warmup_iters = config.warmup_iters
        decay_iters = config.decay_iters
        if warmup_iters > 0:
            warmup_scheduler = lr_scheduler.LinearLR(optimizer,
                                                     start_factor=1e-5,
                                                     total_iters=warmup_iters)
        decay_scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=decay_iters)
        if warmup_iters == 0:
            scheduler = decay_scheduler
        else:
            schedulers = [warmup_scheduler, decay_scheduler]
            scheduler = lr_scheduler.SequentialLR(optimizer,
                                                  schedulers=schedulers,
                                                  milestones=[warmup_iters])
    elif config.scheduler == 'linear_step':
        # Drop lr by 10 at each milestone
        milestones = [int(x) for x in config.milestones.split(',')]
        warmup_iters = config.warmup_iters
        assert warmup_iters > 0

        def lr_lambda(it):
            return 0.1 ** (sum(it >= np.array(milestones)-warmup_iters))
        step_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        warmup_scheduler = lr_scheduler.LinearLR(optimizer,
                                                 start_factor=1e-5,
                                                 total_iters=warmup_iters)
        schedulers = [warmup_scheduler, step_scheduler]
        scheduler = lr_scheduler.SequentialLR(optimizer,
                                              schedulers=schedulers,
                                              milestones=[warmup_iters])
    elif config.scheduler is not None:
        raise ValueError('invalid scheduler %r' % config.scheduler)
    else:
        scheduler = None
    return scheduler


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)


def save_model_helper(epoch, config, tmp=False):
    # default path: ckpts/dataset/model/init/seed/model.pt
    init = 'diff_init' if config.diff_init else 'same_init'

    if config.ckpt_dir is None and config.save_dir is not None:
        # custom directory
        folder_dir = config.save_dir
    else:
        # default directory
        od_suffix = '_od' if config.with_od else ''
        folder_dir = config.ckpt_dir + f'/{config.dataset}' +\
            f'/{config.model}/{init}/seed_{config.seed}{od_suffix}'
    if not tmp:
        name = f'/model_1_{epoch}.pt'
        name_cy = f'/model_2_{epoch}.pt'
    else:
        name = '/model_1_last.pt'
        name_cy = '/model_2_last.pt'
    os.makedirs(folder_dir, exist_ok=True)
    path = folder_dir + name
    path_cy = folder_dir + name_cy
    return folder_dir, path, path_cy
