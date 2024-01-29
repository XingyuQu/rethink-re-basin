import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import random
from enum import Enum

from .logger import Logger
from ..models import cifar_vgg, cifar_resnet, fixup_cifar_resnet, plain_cifar_resnet


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


def load_model(config, in_channels=3):
    if config.dataset in ['cifar10', 'mnist']:
        num_classes = 10
    elif config.dataset == 'cifar100':
        num_classes = 100

    if config.model.startswith('cifar_vgg'):
        last = config.model.split('_')[-1]

        if last.endswith('x'):
            width_multiplier = int(last[:-1])
            model_f = getattr(cifar_vgg, config.model[:-len(last)-1])
        else:
            width_multiplier = 1
            model_f = getattr(cifar_vgg, config.model)
        model = model_f(num_classes=num_classes,
                        special_init=config.special_init,
                        width_multiplier=width_multiplier)

    elif config.model.startswith('cifar_resnet'):
        model_name = config.model
        model = cifar_resnet.ResNet.get_model_from_name(model_name,
                                                        num_classes,
                                                        config.special_init)
    elif config.model.startswith('fixup_cifar_resnet'):
        model_name = config.model
        model = fixup_cifar_resnet.ResNet.get_model_from_name(model_name,
                                                              num_classes,
                                                              config.special_init)
    elif config.model.startswith('plain_cifar_resnet'):
        model_name = config.model
        model = plain_cifar_resnet.ResNet.get_model_from_name(model_name,
                                                              num_classes,
                                                              config.special_init)
    else:
        raise ValueError('invalid model %r' % config.model)

    return model


def load_optimizer(parameters, config):
    if config.optimizer == 'sgd':
        optimizer = optim.SGD(parameters, lr=config.lr,
                              momentum=config.momentum,
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
        folder_dir = config.ckpt_dir + f'/{config.dataset}' +\
            f'/{config.model}/{init}/seed_{config.seed}'
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
