import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import wandb
from source.utils.data_funcs import load_data
from source.utils.utils import load_model, load_optimizer, \
    load_scheduler, seed_everything, save_model_helper
from source.utils.connect import eval_line, calculate_models_distance
from source.utils.logger import Logger
from source.utils.opts import parse_args
from source.utils.calculate_barrier import calculate_barrier
from source.utils import train
from copy import deepcopy
import os
import json


def main():
    """
        Train two models and test lmc along training.
        Record running and lmc statistics.
    """
    args = parse_args()
    # save args
    if args.save_model:
        folder_dir, _, _ = save_model_helper(0, args)
        assert not os.path.exists(f'{folder_dir}/lmc_results.pt')
        with open(f'{folder_dir}/config', 'w') as json_file:
            json.dump(vars(args), json_file, indent=4)
    # Set run name
    if args.run_name is not None:
        run_name = args.run_name
    else:
        run_name = 'train'
        run_name = f'{args.model} {run_name}'
    # Initialize wandb run
    wandb.init(project=args.project, config=args,
               name=run_name, mode=args.wandb_mode,
               notes=args.run_notes)
    config = wandb.config
    # Set random seed
    if config.seed is not None:
        seed = config.seed
        seed_everything(seed)
    # Set device
    if config.device is not None:
        device = torch.device(config.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config.update({'device': device}, allow_val_change=True)
    # create models
    in_channels = 1 if config.dataset == 'mnist' else 3
    model = load_model(config, in_channels)
    model.to(device)
    optimizer = load_optimizer(model.parameters(), config)
    # the other model
    if config.diff_init:
        model_cy = load_model(config, in_channels)
    else:
        model_cy = deepcopy(model)
    model_cy = model_cy.to(device)
    optimizer_cy = load_optimizer(model_cy.parameters(), config)
    # create criterion
    criterion = nn.CrossEntropyLoss() if config.model != 'simple_mlp'\
        else nn.MSELoss()
    # load pretrained model
    if config.init_model:
        path_1 = config.init_model_path_1
        path_2 = config.init_model_path_2
        model.load_state_dict(torch.load(path_1, map_location=device))
        model_cy.load_state_dict(torch.load(path_2, map_location=device))
        config.update({'special_init': 'pretrained'}, allow_val_change=True)
    # Load data
    trainset, testset = load_data(config.data_dir, config.dataset,
                                  no_random_aug=config.no_random_aug)
    # use subset of data for debugging
    if config.subset:
        trainset = torch.utils.data.Subset(trainset, range(20))
        testset = torch.utils.data.Subset(testset, range(20))
    batch_size = config.batch_size
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    trainloader_cy = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # load lr scheduler
    scheduler = load_scheduler(optimizer, config)
    scheduler_cy = load_scheduler(optimizer_cy, config)
    # reset bn statistics when testing lmc
    reset_bn = config.reset_bn
    repair = config.repair
    bn_loader = trainloader if reset_bn or repair else None

    # metrics
    metrics = ['loss', 'top1', 'top5']
    if not config.train_only:
        # running pct for testing lmc
        # should have: lmc_freq * epochs >= 1
        assert config.lmc_freq * config.epochs >= 1
        record_pcts = np.arange(0, 1, config.lmc_freq)
        record_pcts = np.append(record_pcts, 1.0)
        decimal_places = len(str(config.lmc_freq).split('.')[-1])
        record_pcts = np.round(record_pcts, decimal_places)
        record_pt = 0  # pointer to next record_pct
        # Customize x axis
        for cat in ['train', 'test']:
            for metric in metrics:
                wandb.define_metric(f'barrier_{cat}_{metric}',
                                    step_metric='record_pct')
                for record_pct in record_pcts:
                    wandb.define_metric(f'lmc_{record_pct}_{cat}_{metric}',
                                        step_metric='lmc_step')
                for suffix in [1, 2]:
                    wandb.define_metric(f'{cat}_{metric}_{suffix}',
                                        step_metric='epoch')

    wandb.define_metric('distance', step_metric='epoch')
    wandb.define_metric('distance_orign_1', step_metric='epoch')
    wandb.define_metric('distance_orign_2', step_metric='epoch')

    # Train
    # store original model to calulate distance
    model_ori = deepcopy(model)
    model_ori_cy = deepcopy(model_cy)

    epochs = config.epochs
    train_func = train.train
    # results to save
    lmc_results = {'train': [],
                   'test': []}
    running_results = {'train_1': [[], [], []],  # loss, top1, top5
                       'test_1': [[], [], []],
                       'train_2': [[], [], []],
                       'test_2': [[], [], []],
                       'distance': [],
                       'distance_orign_1': [],
                       'distance_orign_2': []}
    for epoch in range(epochs+1):
        # save model at first/last epoch and every save_freq epoch
        save = config.save_model and (epoch % config.save_freq == 0 or
                                      epoch == epochs)
        # we won't save model at epoch 0 if we use pretrained model
        if save and (epoch != 0 or not config.init_model):
            folder_dir, path, path_cy = save_model_helper(epoch, config)
            assert not os.path.exists(f'{folder_dir}/lmc_results.pt')
            sd = model.state_dict()
            sd_cy = model_cy.state_dict()
            torch.save(sd, path)
            torch.save(sd_cy, path_cy)
        # save last ckpt
        if config.save_model:
            folder_dir, path, path_cy = save_model_helper(epoch, config,
                                                          tmp=True)
            assert not os.path.exists(f'{folder_dir}/lmc_results.pt')
            sd = model.state_dict()
            sd_cy = model_cy.state_dict()
            torch.save(sd, path)
            torch.save(sd_cy, path_cy)

        # record lmc statistics if not in training mode
        if not config.train_only:
            record_pct = record_pcts[record_pt]
            # Test lmc
            if epoch == int(epochs*record_pct):
                # Record barriers
                n = config.n  # Number of test points on the line segment
                lmc_train_stat = eval_line(
                    model, model_cy, trainloader, criterion, device, config,
                    n=n, reset_bn=reset_bn, bn_loader=bn_loader,
                    repair=config.repair)
                lmc_test_stat = eval_line(
                    model, model_cy, testloader, criterion, device, config,
                    n=n, reset_bn=reset_bn, bn_loader=bn_loader,
                    repair=config.repair)
                lmc_results['train'].append(lmc_train_stat)
                lmc_results['test'].append(lmc_test_stat)

                barriers_train = [calculate_barrier(
                    lmc_train_stat[:, col], metric)
                                 for col, metric in enumerate(metrics)]
                barriers_test = [calculate_barrier(
                    lmc_test_stat[:, col], metric)
                                for col, metric in enumerate(metrics)]

                log_dict = {f'barrier_train_{metric}': barriers_train[
                    metrics.index(metric)] for metric in metrics}
                log_dict.update({f'barrier_test_{metric}': barriers_test[
                    metrics.index(metric)] for metric in metrics})
                log_dict.update({'record_pct': record_pct})
                wandb.log(log_dict)

                # Record all lmc statistics
                lmc_steps = np.round(np.linspace(0, 1, n), 2)
                for cat, lmc_stat in zip(('train', 'test'), (lmc_train_stat,
                                                             lmc_test_stat)):
                    for metric in metrics:
                        y = lmc_stat[:, metrics.index(metric)]
                        # log data
                        metric_name = f'lmc_{record_pct}_{cat}_{metric}'
                        for i in range(n):
                            wandb.log({metric_name: y[i],
                                       'lmc_step': lmc_steps[i]})

                record_pt += 1

        # Validate models on test set
        if epoch % config.test_freq == 0 or epoch == epochs:
            loss_1, top1_1, top5_1, _ = train.validate(
                testloader, model, criterion, device, config)
            loss_2, top1_2, top5_2, _ = train.validate(
                testloader, model_cy, criterion, device, config)
            var_names = ['loss', 'top1', 'top5']
            for i in range(3):
                running_results['test_1'][i].append(eval(var_names[i]+'_1'))
                running_results['test_2'][i].append(eval(var_names[i]+'_2'))
            for suffix in range(1, 3):
                wandb.log(
                    {f'test_loss_{suffix}': eval(f'loss_{suffix}'),
                     f'test_top1_{suffix}': eval(f'top1_{suffix}'),
                     f'test_top5_{suffix}': eval(f'top5_{suffix}'),
                     'epoch': epoch})

        # measure training statistics at epoch 0
        if epoch == 0:
            loader = trainloader
            loader_cy = trainloader_cy
            # unpack (target, output) in validate_func
            validate_func = train.validate
            loss_1, top1_1, top5_1, _ = validate_func(
                loader, model, criterion, device, config)
            loss_2, top1_2, top5_2, _ = validate_func(
                loader_cy, model_cy, criterion, device, config)
            var_names = ['loss', 'top1', 'top5']
            for i in range(3):
                running_results['train_1'][i].append(eval(var_names[i]+'_1'))
                running_results['train_2'][i].append(eval(var_names[i]+'_2'))
            for suffix in range(1, 3):
                wandb.log(
                    {f'train_loss_{suffix}': eval(f'loss_{suffix}'),
                     f'train_top1_{suffix}': eval(f'top1_{suffix}'),
                     f'train_top5_{suffix}': eval(f'top5_{suffix}'),
                     'epoch': epoch})

        # Record distance between models
        distance = calculate_models_distance(model, model_cy)
        distance_orign_1 = calculate_models_distance(model_ori, model)
        distance_orign_2 = calculate_models_distance(model_ori_cy, model_cy)
        wandb.log({'distance': distance,
                   'distance_orign_1': distance_orign_1,
                   'distance_orign_2': distance_orign_2,
                   'epoch': epoch})
        running_results['distance'].append(distance)
        running_results['distance_orign_1'].append(distance_orign_1)
        running_results['distance_orign_2'].append(distance_orign_2)

        # Only record $epochs epochs results
        if epoch == epochs:
            break

        # Train models
        loader = trainloader
        loader_cy = trainloader_cy
        loss_1, top1_1, top5_1, _ = train_func(
            loader, model, criterion, optimizer,
            epoch, device, config, scheduler)
        loss_2, top1_2, top5_2, _ = train_func(
            loader_cy, model_cy, criterion, optimizer_cy,
            epoch, device, config, scheduler_cy)
        log_dict = {'epoch': epoch+1}
        var_names = ['loss', 'top1', 'top5']
        for i in range(3):
            running_results['train_1'][i].append(eval(var_names[i]+'_1'))
            running_results['train_2'][i].append(eval(var_names[i]+'_2'))
        for suffix in range(1, 3):
            log_dict.update({f'train_loss_{suffix}': eval(f'loss_{suffix}'),
                             f'train_top1_{suffix}': eval(f'top1_{suffix}'),
                             f'train_top5_{suffix}': eval(f'top5_{suffix}')})
        wandb.log(log_dict)

    # Save results
    torch.save(lmc_results, f'{folder_dir}/lmc_results.pt')
    torch.save(running_results, f'{folder_dir}/running_results.pt')

    os.remove(f'{folder_dir}/model_1_last.pt')
    os.remove(f'{folder_dir}/model_2_last.pt')


if __name__ == '__main__':
    Logger.setup_logging()
    logger = Logger()
    wandb.login()
    main()
