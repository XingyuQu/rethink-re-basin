import time

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from source.utils.utils import AverageMeter, ProgressMeter, \
    Summary, accuracy


def train(train_loader, model, criterion, optimizer,
          epoch, device, config, scheduler=None, iter=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data, target) in enumerate(train_loader):
        if iter is not None and i >= iter:
            break
        # measure data loading time
        data_time.update(time.time() - end)

        data = data.to(device)
        target = target.to(device)

        # compute output
        output_full = model(data)
        loss = criterion(output_full, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output_full, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0].item(), data.size(0))
        top5.update(acc5[0].item(), data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            progress.display(i + 1)
    return losses.avg, top1.avg, top5.avg, batch_time.avg


def validate(val_loader, model, criterion, device, config):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (data, target) in enumerate(loader):
                i = base_progress + i
                data = data.to(device)
                target = target.to(device)

                # compute output
                output = model(data)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), data.size(0))
                top1.update(acc1[0].item(), data.size(0))
                top5.update(acc5[0].item(), data.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % config.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    if config.model == 'simple_mlp':
        criterion = nn.MSELoss()

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)

    progress.display_summary()

    return losses.avg, top1.avg, top5.avg, batch_time.avg
