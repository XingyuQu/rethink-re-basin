import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from source.utils.data_funcs import load_data
from source.utils.utils import load_model, load_optimizer, \
    load_scheduler, seed_everything
from source.utils.connect import calculate_models_distance
from source.utils.logger import Logger
from source.utils.opts import parse_args
from source.utils import train
from copy import deepcopy
import os
import json


def main():
    """
        Train a single model.
        Record running statistics.
    """
    args = parse_args()
    # save args
    if args.save_model:
        folder_dir = args.save_dir
        os.makedirs(folder_dir, exist_ok=True)
        assert not os.path.exists(f'{folder_dir}/running_results.pt')
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
    # load pretrained model
    if config.init_model:
        path_1 = config.init_model_path_1
        model.load_state_dict(torch.load(path_1, map_location=device))
        config.update({'special_init': 'pretrained'}, allow_val_change=True)
    optimizer = load_optimizer(model.parameters(), config)

    # create criterion
    criterion = nn.CrossEntropyLoss() if config.model != 'simple_mlp'\
        else nn.MSELoss()
    # Load data
    trainset, testset = load_data(config.data_dir, config.dataset,
                                  no_random_aug=config.no_random_aug)
    # use subset of data for debugging
    if config.subset:
        trainset = torch.utils.data.Subset(trainset, range(20))
        testset = torch.utils.data.Subset(testset, range(20))
    batch_size = config.batch_size
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # load lr scheduler
    scheduler = load_scheduler(optimizer, config)

    # Customize x axis
    metrics = ['loss', 'top1', 'top5']
    for cat in ['train', 'test']:
        for metric in metrics:
            for suffix in [1]:
                wandb.define_metric(f'{cat}_{metric}_{suffix}',
                                    step_metric='epoch')
    wandb.define_metric('distance_orign_1', step_metric='epoch')

    # Train
    # store original model to calulate distance
    model_ori = deepcopy(model)

    epochs = config.epochs
    train_func = train.train
    # results to save
    running_results = {'train_1': [[], [], []],  # loss, top1, top5
                       'test_1': [[], [], []]}
    for epoch in range(epochs+1):
        # save model at first/last epoch and every save_freq epoch
        save = config.save_model and (epoch % config.save_freq == 0 or
                                      epoch == epochs)
        # we won't save model at epoch 0 if we use pretrained model
        if save and (epoch != 0 or not config.init_model):
            assert not os.path.exists(f'{folder_dir}/running_results.pt')
            path = f'{folder_dir}/model_1_{epoch}.pt'
            sd = model.state_dict()
            torch.save(sd, path)
        # save last ckpt
        if config.save_model:
            assert not os.path.exists(f'{folder_dir}/lmc_results.pt')
            path = f'{folder_dir}/model_1_last.pt'
            sd = model.state_dict()
            torch.save(sd, path)

        # Validate models on test set
        if epoch % config.test_freq == 0 or epoch == epochs:
            loss_1, top1_1, top5_1, _ = train.validate(
                testloader, model, criterion, device, config)
            var_names = ['loss', 'top1', 'top5']
            for i in range(3):
                running_results['test_1'][i].append(eval(var_names[i]+'_1'))
            for suffix in range(1, 2):
                wandb.log(
                    {f'test_loss_{suffix}': eval(f'loss_{suffix}'),
                     f'test_top1_{suffix}': eval(f'top1_{suffix}'),
                     f'test_top5_{suffix}': eval(f'top5_{suffix}'),
                     'epoch': epoch})

        # measure training statistics at epoch 0
        if epoch == 0:
            loader = trainloader
            # unpack (target, output) in validate_func
            validate_func = train.validate
            loss_1, top1_1, top5_1, _ = validate_func(
                loader, model, criterion, device, config)
            var_names = ['loss', 'top1', 'top5']
            for i in range(3):
                running_results['train_1'][i].append(eval(var_names[i]+'_1'))
            for suffix in range(1, 2):
                wandb.log({'train_loss_1': eval(f'loss_{suffix}'),
                           'train_top1_1': eval(f'top1_{suffix}'),
                           'train_top5_1': eval(f'top5_{suffix}'),
                           'epoch': epoch})

        # Record distance between models
        wandb.log({'distance_orign_1': calculate_models_distance(
                       model_ori, model),
                   'epoch': epoch})

        # Only record $epochs epochs results
        if epoch == epochs:
            break

        # Train models
        loader = trainloader
        loss_1, top1_1, top5_1, _ = train_func(
            loader, model, criterion, optimizer,
            epoch, device, config, scheduler)
        log_dict = {'epoch': epoch+1}
        var_names = ['loss', 'top1', 'top5']
        for i in range(3):
            running_results['train_1'][i].append(eval(var_names[i]+'_1'))
        for suffix in range(1, 2):
            log_dict.update({f'train_loss_{suffix}': eval(f'loss_{suffix}'),
                             f'train_top1_{suffix}': eval(f'top1_{suffix}'),
                             f'train_top5_{suffix}': eval(f'top5_{suffix}')})
        wandb.log(log_dict)

    # Save results
    torch.save(running_results, f'{folder_dir}/running_results.pt')
    os.remove(f'{folder_dir}/model_1_last.pt')


if __name__ == '__main__':
    Logger.setup_logging()
    logger = Logger()
    wandb.login()
    main()
