
"""
Managing class for different Policies.
"""

from models import get_model
from policies.pruners import build_pruners_from_config
from policies.trainers import build_training_policy_from_config
from policies.recyclers import build_recyclers_from_config
from policies.regularizers import build_regs_from_config

from torch.nn import ReLU



from utils.utils import (preprocess_for_device, recompute_bn_stats,
                        TrainingProgressTracker, normalize_module_name, get_total_sparsity,
                        get_total_sparsity_unwrapped)

from utils.checkpoints import load_checkpoint, save_checkpoint, get_unwrapped_model
from utils.flop_utils import get_flops, get_macs_dpf

from utils.csv_writer import save_experiment_results, save_epoch_results
from utils.datasets import get_datasets
from utils.manager_utils import (compare_models, my_test, my_test_dataset, save_plot,
                                 get_dataloaders, _init_fn, get_linear_conv_module_names, analyse_loss_path)
from utils.masking_utils import get_wrapped_model
from utils.parse_config import read_config, write_config

import math
import torch
import torch.onnx 
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import os
import logging
import copy
import random
import sys

import lmc_source


USE_TQDM = True
if not USE_TQDM:
    tqdm = lambda x: x

DEFAULT_TRAINER_NAME = "default_trainer"
'''
trainers:
  default_trainer:
    optimizer:
      class: SGD
      lr: 0.1
    lr_scheduler:
This is how the trainer config looks like on the based of which 
optimizer, scheduler and 
'''

class Manager:
    """
    Class for invoking trainers and pruners.
    """
    def __init__(self, args):
        self.seed = args.seed
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        args = preprocess_for_device(args)

        if args.deterministic:
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if args.num_path_steps > 0:
            if not args.prune_direction:
                logging.info('setting prune direction to True!')
                args.prune_direction = True

        self.use_aa = args.aa

        logging.info("Also printing results@top5 as well!")
        args.report_top5 = True

        self.model_config = {'arch': args.arch, 'dataset': args.dset, 'use_butterfly': args.use_butterfly}

        self.model = get_model(args.arch, args.dset, args.pretrained, args.use_butterfly, args.use_se, args.se_ratio, args.kernel_sizes, args.p, args=args)

        self.config = args.config_path if isinstance(args.config_path, dict) else read_config(args.config_path)
        logging.info(f"this is config before {self.config}")
        if args.update_config:
            self.online_update_config(args)
            logging.info(f"this is config after {self.config}")

        layers = []
        logging.info("=======Layers:=============")
        for name, layer in self.model.named_modules():
            if isinstance(layer, ReLU):
                layers.append(name)
                logging.info('         ' + str(name) + ',')

        logging.info("====================")

        self.data = (args.dset, args.dset_path)

        self.n_epochs = args.epochs
        logging.info(f'total epochs is {self.n_epochs}')
        self.num_workers = args.workers
        self.batch_size = args.batch_size
        if 'woodfisher' in args.prune_class or 'woodtaylor' in args.prune_class or 'grad' in args.prune_class or 'kfac' in args.prune_class:
            if not hasattr(args, 'fisher_mini_bsz'):
                args.fisher_mini_bsz = 1
            if not hasattr(args, 'max_mini_bsz') or args.max_mini_bsz is None:
                args.num_samples = args.fisher_subsample_size * args.fisher_mini_bsz
            else:
                args.num_samples = args.fisher_subsample_size * int(math.ceil(args.fisher_mini_bsz/args.max_mini_bsz)) * args.max_mini_bsz
                logging.info(f'Setting number of samples to {args.num_samples} when max_mini_bsz is enabled')

            if 'woodtaylor' in args.prune_class and args.grad_subsample_size is not None:
                assert args.grad_subsample_size >= args.fisher_subsample_size * args.fisher_mini_bsz
                assert args.grad_subsample_size % args.fisher_mini_bsz == 0
                logging.info(f'For Taylor: setting num_samples (for grads) from {args.num_samples} to {args.grad_subsample_size}')
                args.num_samples = args.grad_subsample_size

        self.num_samples = args.num_samples
        self.device = args.device
        self.initial_epoch = 1
        self.best_val_acc = 0.

        self.recompute_bn_stats = args.recompute_bn_stats
        self.training_stats_freq = args.training_stats_freq

        if args.from_checkpoint_path is not None:
            if args.use_model_config:
                logging.info(f"{args.not_oldfashioned}, not_oldfashioned")
                args.old_fashioned = not args.not_oldfashioned

                # can rely on the build_training_policy
                epoch, model, _, _ = load_checkpoint(args.from_checkpoint_path, self.model_config,
                                            self.config, DEFAULT_TRAINER_NAME, args.old_fashioned, args.ckpt_epoch, args=args)
            else:
                args.old_fashioned = False
                epoch, model, _, _ = load_checkpoint(args.from_checkpoint_path, args=args)

            self.model = model

            if args.reset_training_policy:
                raise NotImplementedError
        else:
            if not args.export_onnx:
                self.model = get_wrapped_model(self.model)

        if args.device.type == 'cuda':
            # if len(args.gpus) > 1:
            if not args.no_dataparallel:
                self.model = torch.nn.DataParallel(self.model, device_ids=args.gpus)
                self.model.to(args.device)
            else:
                self.model = self.model.to(args.device)

        self.pruners = build_pruners_from_config(self.model, self.config, inp_args=args)
        self.recyclers = build_recyclers_from_config(self.model, self.config)
        self.regularizers = build_regs_from_config(self.model, self.config)

        self.trainers = [build_training_policy_from_config(self.model, self.config, trainer_name=DEFAULT_TRAINER_NAME, label_smoothing=args.label_smoothing)]

        self.setup_logging(args)
        logging.info(f'initial sparsity in args is {args.init_sparsity}')
        logging.info(f"initial sparsity in config is {self.config['pruners']['pruner_1']['initial_sparsity']}")
        self.args = args

        if args.mask_onnx:
            if args.eval_fast:
                train_loader, test_loader = get_dataloaders(args, self.data)
                init_acc_top1, init_loss, _ = my_test(args, self.model, log_dict=None, test_loader=test_loader)
                logging.info("The init acc and loss are ", init_acc_top1, init_loss)

            device_before = next(self.model.named_parameters())[1].device
            self.model.to('cpu')
            self.model = get_wrapped_model(self.model)

            self.run_policies_for_method('pruner',
                                         'after_parameter_optimization',
                                         model=self.model)

            self.model = get_unwrapped_model(self.model)
            self.model.to(device_before)

            sparsity_dicts = self.run_policies_for_method('pruner',
                                                          'on_epoch_end')

            if args.save_dense_also:
                save_checkpoint(epoch, self.model_config, self.model, self.trainers[0].optimizer,
                                self.trainers[0].lr_scheduler, self.run_dir, is_best=False, nick='sparsified')

        if args.export_onnx is True:

            x = torch.randn(args.batch_size, 3, 224, 224, requires_grad=True).to(args.device)
                #torch_out = self.model(x)

            if args.onnx_nick:
                onnx_nick = args.onnx_nick
            else:
                onnx_nick = 'resnet_pruned.onnx'

                # Export the model
            torch.onnx.export(self.model.module,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  onnx_nick,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'] )
            logging.info("ONNX EXPORT COMPLETED. EXITING")
            sys.exit()

    def setup_logging(self, args):
        self.logging_level = args.logging_level
        self.checkpoint_freq = args.checkpoint_freq
        self.exp_dir = args.exp_dir
        self.run_dir = args.run_dir

    def _prune_class_nick_to_fullname(self, name):

        dic = {'magni': 'MagnitudePruner',
               'globalmagni': 'GlobalMagnitudePruner',
               'diagfisher': 'FisherPruner',
               'naivehess': 'NaiveHessianPruner',
               'woodfisher': 'WoodburryFisherPruner',
               'woodtaylor': 'WoodburryTaylorPruner',
               'woodfisherblock': 'BlockwiseWoodburryFisherPruner',
               'woodfisherblock_flops': 'FlopsBlockwiseWoodburryFisherPruner',
               'woodtaylorblock': 'BlockwiseWoodburryTaylorPruner',
               'kfac': 'KFACFisherPruner',
            }

        if name not in dic:
            raise NotImplementedError
        else:
            return dic[name]

    def online_update_config(self, args):

        # PR: Extend handling of this method to multiple pruners or a hybrid of pruners

        self.backup_config = copy.deepcopy(self.config)

        for pruner in self.config['pruners']:
            self.config['pruners'][pruner]['class'] = self._prune_class_nick_to_fullname(args.prune_class)
            self.config['pruners'][pruner]['weight_only'] = not args.prune_bias

        if args.prune_modules is not None:
            self.config['pruners']['pruner_1']['modules'] = args.prune_modules.split('_')

        if args.prune_all:
            self.config['pruners']['pruner_1']['modules'] = get_linear_conv_module_names(self.model)

        if args.init_sparsity is not None:
            self.config['pruners']['pruner_1']['initial_sparsity'] = args.init_sparsity
        else:
            # update the args!
            if 'init_sparsity' in self.config['pruners']['pruner_1'] and self.config['pruners']['pruner_1']['initial_sparsity'] is not None:
                setattr(args, 'init_sparsity', self.config['pruners']['pruner_1']['initial_sparsity'])
                setattr(self.args, 'init_sparsity', self.config['pruners']['pruner_1']['initial_sparsity'])

        if args.target_sparsity is not None:
            self.config['pruners']['pruner_1']['target_sparsity'] = args.target_sparsity

        if args.one_shot:
            logging.info('Overwrite arguments for one-shot!')
            args.epochs = 1 # only 1 epoch of pruning proces!
            # assert args.prune_end != 1 # prune active method doesn't support it yet!
            assert args.target_sparsity is not None
            if args.init_sparsity is None:
                args.init_sparsity = args.target_sparsity
                for pruner in self.config['pruners']:
                    self.config['pruners'][pruner]['initial_sparsity'] = args.target_sparsity
            else:
                assert args.target_sparsity == args.init_sparsity

            args.prune_start = 0
            args.prune_freq = 1
            args.prune_end = 1

            for pruner in self.config['pruners']:
                self.config['pruners'][pruner]['epochs'] = [args.prune_start, args.prune_freq, args.prune_end]

        else:
            if args.prune_start is not None and args.prune_end is not None and args.prune_freq is not None:
                # if all provided in args, override everything
                self.config['pruners']['pruner_1']['epochs'] = [args.prune_start, args.prune_freq, args.prune_end]
            elif args.prune_start is None and args.prune_end is not None and args.prune_freq is not None:
                # if end and freq are provided in args, assume start as 0
                self.config['pruners']['pruner_1']['epochs'] = [0, args.prune_freq, args.prune_end]
                args.prune_start = 0
            else:
                # take care of the rest of possibilities
                if args.prune_start is not None:
                    self.config['pruners']['pruner_1']['epochs'][0] = args.prune_start
                else:
                    args.prune_start = self.config['pruners']['pruner_1']['epochs'][0]

                if args.prune_freq is not None:
                    self.config['pruners']['pruner_1']['epochs'][1] = args.prune_freq
                else:
                    args.prune_freq = self.config['pruners']['pruner_1']['epochs'][1]

                if args.prune_end is not None:
                    self.config['pruners']['pruner_1']['epochs'][2] = args.prune_end
                else:
                    args.prune_end = self.config['pruners']['pruner_1']['epochs'][2]

        if not hasattr(args, 'epochs') or args.epochs is None:
            args.epochs = args.prune_end + 1
        elif args.prune_end is not None and args.epochs <= args.prune_end and not args.one_shot:
            args.epochs = args.prune_end + 1

        if hasattr(args, 'prune_optimizer') and args.prune_optimizer is not None:
            self.config['trainers']['default_trainer']['optimizer']['class'] = args.prune_optimizer

        if args.prune_lr is not None:
            self.config['trainers']['default_trainer']['optimizer']['lr'] = args.prune_lr

        if args.prune_momentum is not None:
            if 'momentum' in self.config['trainers']['default_trainer']['optimizer']:
                self.config['trainers']['default_trainer']['optimizer']['momentum'] = args.prune_momentum
            elif args.set_prune_momentum and 'momentum' not in self.config['trainers']['default_trainer']['optimizer']:
                self.config['trainers']['default_trainer']['optimizer']['momentum'] = args.prune_momentum

        if args.prune_wdecay is not None:
            # update weight decay in config
            self.config['trainers']['default_trainer']['optimizer']['weight_decay'] = args.prune_wdecay
        else:
            if 'weight_decay' in self.config['trainers']['default_trainer']['optimizer']:
                # save weight decay value into args
                args.prune_wdecay = self.config['trainers']['default_trainer']['optimizer']['weight_decay']
            else:
                args.prune_wdecay = 0

    def run_policies_for_method(self, policy_type: str, method_name: str, **kwargs):

        if 'agg_func' not in kwargs:
            agg_func = None
        else:
            agg_func = kwargs['agg_func']
            del kwargs['agg_func']
        res = []
        for policy_obj in getattr(self, f"{policy_type}s"):
            retval = getattr(policy_obj, method_name)(**kwargs)
            res.append(retval)
        # logging.info(f'print res: {res}')
        if policy_type == 'pruner' and method_name == 'on_epoch_begin':
            is_active, meta = [r[0] for r in res], [r[1] for r in res]
            if any(is_active) and self.recompute_bn_stats:
                dataloader = DataLoader(
                    get_datasets(*self.data, train_random_transforms=not self.args.disable_train_random_transforms)[0],
                    batch_size=self.batch_size, shuffle=not self.args.disable_train_shuffle,
                )

                recompute_bn_stats(self.model, dataloader, self.device)
            # assert len(res) == 1
            meta[0]['is_active'] = is_active[0]
            return meta
        return res if agg_func is None else agg_func(res)


    def summarize(self, epoch, test_loader):
        sparsity_dicts = self.run_policies_for_method('pruner', 
                                                      'on_epoch_end')
        logging.info(sparsity_dicts)
        processed_sparsity_dict = self.training_progress.sparsity_info(epoch,
                                             sparsity_dicts, 
                                             *get_total_sparsity(self.model))

        val_loss, val_correct_top1, val_correct_top5 = self.run_policies_for_method('trainer',
                                                            'on_epoch_end',
                                                            dataloader=test_loader,
                                                            device=self.device,
                                                            epoch_num=epoch,
                                                            agg_func=lambda x: np.sum(x, axis=0))
        self.training_progress.val_info(epoch, val_loss, val_correct_top1)
        return val_correct_top1, processed_sparsity_dict, val_correct_top5

    def run(self):
        print("The args are : ", self.args)
        data_train, data_test = get_datasets(*self.data, use_aa=self.use_aa, train_random_transforms=not self.args.disable_train_random_transforms)
        train_loader = DataLoader(data_train, batch_size=self.batch_size, shuffle=not self.args.disable_train_shuffle, num_workers=self.num_workers, worker_init_fn=_init_fn)
        test_loader = DataLoader(data_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, worker_init_fn=_init_fn)

        self.training_progress = TrainingProgressTracker(self.initial_epoch,
                                                         len(train_loader),
                                                         len(test_loader.dataset),
                                                         self.training_stats_freq,
                                                         self.run_dir)

        logging.info('====>Initial summary before doing any training/pruning :')
        results_top1 = []
        results_top5 = []
        logging.info("doing my custom test on test set")
        st_time = time.perf_counter()
        
        # 选择你要查询的GPU（如果有多个GPU的话）
        # gpu_id = 0  # 选择GPU 0
        # gpu_properties = torch.cuda.get_device_properties(gpu_id)

        # print(f"GPU Name: {gpu_properties.name}")
        # print(f"Total GPU Memory: {gpu_properties.total_memory / (1024**3):.2f} GB")
        # print(f"Allocated GPU Memory: {torch.cuda.memory_allocated(gpu_id) / (1024**3):.2f} GB")
        # print(f"Free GPU Memory: {torch.cuda.memory_reserved(gpu_id) / (1024**3):.2f} GB")
            
        # if self.args.eval_fast:
        #     init_acc_top1, init_loss, init_acc_top5 = my_test(self.args, self.model, log_dict=None, test_loader=test_loader)
        # else:
        #     init_acc_top1, init_loss, init_acc_top5 = my_test_dataset(self.args, self.model, data_test, batched=self.args.batched_test, device=self.device)

        # self.training_progress.tensorboard_info(0, init_loss, init_acc_top1)
        
        init_acc_top1 = None
        init_acc_top5 = None

        end_time = time.perf_counter()
        if self.args.flops:
            logging.info('Getting MACs of this model')
            logging.info(get_macs_dpf(self.args, self.model))

        logging.info("Time taken for my_test_dataset in batched={} mode before for loop is {} seconds".format(self.args.batched_test, str(end_time - st_time)))
        results_top1.append(init_acc_top1)
        results_top5.append(init_acc_top5)
        logging.info(f"len of test_loader {len(test_loader.dataset)}")

        if self.args.check_train_loss or self.args.prune_direction:
            logging.info("doing my custom test on training set")
            st_time = time.perf_counter()
            init_train_acc, init_train_loss, init_train_accK = my_test_dataset(self.args, self.model, data_train, batched=self.args.batched_test, mode='train', device=self.device)
            end_time = time.perf_counter()
            logging.info("Time taken for my_test_dataset in batched={} mode before for loop is {} seconds".format(
                self.args.batched_test, str(end_time - st_time)))
            logging.info('Train accuracy and loss are respectively: {} and {}'.format(init_train_acc, init_train_loss))


        before_prune_acc_top1 = init_acc_top1
        before_prune_acc_top5 = init_acc_top5
        
        if self.args.cache_subset_ids:
            subset_ids_cached = np.random.choice(len(data_train), len(data_train), replace=False)
            logging.info(f"initially size of subset ids cached is {len(subset_ids_cached)}")
            subset_ids_cached = subset_ids_cached[0:self.num_samples]
        else:
            subset_ids_cached = None


        backup_model = None
        num_pruning_epochs = 0
        best_epoch_idx = -1
        best_epoch_acc = -1
        for epoch in range(self.initial_epoch, self.n_epochs):
            logging.info(f"Starting epoch {epoch} with number of batches {len(train_loader)}")

            epoch_results_top1 = []
            epoch_results_top5 = []
            is_pruning_epoch = False
            epoch_results_top1.append(before_prune_acc_top1) # before prune acc (i.e., top 1)
            epoch_results_top5.append(before_prune_acc_top5)  # before prune accK (i.e., top 5)
            if 'ln' in self.args.arch or self.args.arch in ['my_cifar_vgg16', 'my_plain_cifar_resnet20']:
                logging.info('--------------Store model for REPAIR-----------------')
                model_tmp = copy.deepcopy(self.model)
                
            # logging.info("performance at the start of the epoch is ")
            self.summarize(-1, test_loader)
            if subset_ids_cached is None:
                if self.args.full_subsample:
                    subset_inds = np.random.choice(len(data_train), len(data_train), replace=False)
                    subset_inds = subset_inds[0:self.num_samples]
                else:
                    subset_inds = np.random.choice(len(data_train), self.num_samples, replace=False)
            else:
                logging.info("use cached subset ids! ")
                subset_inds = subset_ids_cached

            st_time = time.perf_counter()
            metas = self.run_policies_for_method('pruner',
                                                'on_epoch_begin',
                                                num_workers=self.num_workers,
                                                dset=data_train,
                                                device=self.device,
                                                subset_inds=subset_inds,
                                                batch_size=64,
                                                epoch_num=epoch)
            end_time = time.perf_counter()
            prune_time = end_time - st_time
            logging.info(f"metas[is_active] is {metas[0]['is_active']}")
            logging.info("performance after pruning (if it was applied)")
            st_time = time.perf_counter()
            if self.args.eval_fast:
                after_prune_acc_top1, after_prune_loss, after_prune_acc_top5 = my_test(self.args, self.model, log_dict=None,
                                                              test_loader=test_loader)
            else:
                after_prune_acc_top1, after_prune_loss, after_prune_acc_top5 = my_test_dataset(self.args, self.model, data_test, batched=self.args.batched_test, device=self.device)

            end_time = time.perf_counter()

            results_top1.append(after_prune_acc_top1) # after prune acc
            results_top5.append(after_prune_acc_top5)  # after prune accK
            
            logging.info("--------------Re-Normalization-----------------")
            import lmc_source
            import lmc_source.utils
            import lmc_source.utils.connect
            from copy import deepcopy
            model_new = deepcopy(self.model)
            if self.args.arch in ['my_cifar_vgg16_bn', 'my_cifar_resnet20', 'resnet50']:
                lmc_source.utils.connect.reset_bn_stats(model_new, self.device, train_loader, self.args.num_samples)
            else:
                repair_type = 'rescale' if self.args.arch == 'my_cifar_vgg16' else 'repair'
                model_new = lmc_source.utils.connect.repair(train_loader, [model_tmp], self.model, self.device, name=self.args.arch, num_samples=self.args.num_samples, variant=repair_type)
            test_acc_top1, test_loss, test_acc_top5 = my_test(self.args, model_new, log_dict=None, test_loader=test_loader)
            logging.info(f'After re-normalization (test acc, loss, top5): {test_acc_top1}, {test_loss}, {test_acc_top5}')
            
            # The above code is declaring a variable named "results_top1".
            results_top1.append(test_acc_top1) # repair prune acc
            results_top5.append(test_acc_top5)  # repair prune accK
            logging.info("--------------Re-Normalization End-----------------")

            logging.info("Time taken for my_test_dataset in batched={} mode after prune is {} seconds".format(self.args.batched_test, str(end_time - st_time)))
            if metas[0]['is_active']:
                num_pruning_epochs += 1
                is_pruning_epoch = True
                setattr(self.args, 'prune_time', prune_time)

                self.training_progress.tensorboard_info(epoch, after_prune_loss, after_prune_acc_top1)

                if self.args.inspect_inv:
                    logging.info(metas[0]['inspect_dic'])
                    for key, val in metas[0]['inspect_dic'].items():
                        setattr(self.args, 'inspect_' + key, val)

            if self.args.check_train_loss or self.args.prune_direction:
                logging.info("(after prune?) doing my custom test on training set")
                st_time = time.perf_counter()
                after_prune_train_acc, after_prune_train_loss, after_prune_train_accK = my_test_dataset(self.args, self.model, data_train,
                                                                  batched=self.args.batched_test, mode='train', device=self.device)
                end_time = time.perf_counter()
                logging.info("Time taken for my_test_dataset in batched={} mode before for loop is {} seconds".format(
                    self.args.batched_test, str(end_time - st_time)))
                logging.info('(After prune?) Train accuracy and loss are respectively: {} and {}'.format(after_prune_train_acc, after_prune_train_loss))

            if metas[0]['is_active'] and self.args.prune_direction:
                assert len(metas) == 1
                self.analyse_loss_path(copy.deepcopy(self.model), metas[0]['prune_direction'], metas[0]['original_param'], metas[0]['mask_previous'], metas[0]['mask_overall'], [data_test, data_train], modes=['test', 'train'], backup_model=backup_model, quad_term = metas[0]['quad_term'])

            self.training_progress.meta_info(epoch, metas)
            self.run_policies_for_method('recycler',
                                         'on_epoch_begin',
                                         num_workers=self.num_workers,
                                         dset=data_train,
                                         device=self.device,
                                         subset_inds=subset_inds,
                                         batch_size=64,
                                         epoch_num=epoch,
                                         optimizer=self.trainers[0].optimizer)

            if epoch == self.args.prune_end and self.args.disable_wdecay_after_prune:
                # PR: note that it might be the case the pruning target has reached but the epoch is less than prune_end
                # due to the particular pruning interval value. in that case, wdecay will remain enabled for those
                # few epochs
                logging.info("Disable weight decay post pruning")
                logging.info(f"Weight decay before: {self.trainers[0].optimizer.param_groups[0]['weight_decay']}")
                self.trainers[0].optimizer.param_groups[0]['weight_decay'] = 0.
                logging.info(f"Weight decay now: {self.trainers[0].optimizer.param_groups[0]['weight_decay']}")
            
            # print('results_top1:', epoch_results_top1)

            val_correct_top1, processed_sparsity_dict, val_correct_top5 = self.summarize(epoch, test_loader)
            
            # save_epoch_results(self.args, self.config, epoch_results_top1, epoch, processed_sparsity_dict, is_pruning_epoch, epoch_results_top5)

        # summarize even if epochs==0
        # logging.info('====>Final summary for the run:')
        # val_correct_top1, processed_sparsity_dict, val_correct_top5 = self.summarize(self.n_epochs, test_loader)
        val_correct_top1 = None
    

        save_experiment_results(self.args, self.config, results_top1, results_top5)

        # get_flops(self.args, self.model)

        # write_config(self.config, os.path.join(self.args.run_dir, 'config.yaml'))

        logging.info("------------------ Summary of results: ------------------ ")
        logging.info(f"results@top1 are {results_top1}")
        logging.info(f"results@top5 are {results_top5}")
        logging.info(f"best accuracy is {best_epoch_acc} at epoch {best_epoch_idx}")

        logging.info(f"The args used were: {self.args}")
        logging.info(f"And the config used was: {self.config}")
        logging.info("----------------------------------------------------------")

        return val_correct_top1, len(test_loader.dataset)
