# Rethink Model Re-Basin and the Linear Mode Connectivity

This repository contains PyTorch implementation for results presented in the paper: *Rethink Model Re-Basin and the Linear Mode Connectivity*.

## Structure of the repo

* `run_ex.py` is the main file for training models. It trains a pair of models simultaneously.
* `run_training.py` is similar to `run_ex.py`, but only trains a single model at a time. It can be utilized for training a model with large width.
* `training_scripts/` contains bash scripts for training models, including all the hyper-parameter settings.
* `source/` contains source code for model, dataset, training, matching, etc.
  * `source/utils/fim.py` is adpted from [fim.py](https://github.com/tudor-berariu/fisher-information-matrix/blob/master/fim.py), contaning code for calculating the Fisher Information.
  * `source/utils/opts.py` contains code for parsing arguments.
  * `source/utils/weight_matching/` contains code for weight matching.
  * `source/utils/activation_matching/` contains code for activation pruning.
  * `source/utils/connect/` contains code for re-normalization and interpolation.
* `notebooks/` contains notebooks to reproduce the results (except some in pruning) presented in the paper. It contains the least code to showcase all the functionalities of the repo and can be easily extended to all the settings.
* `WoodFisher/` is adapted from [WoodFisher](https://github.com/IST-DASLab/WoodFisher) for pruning experiemnts. A complete description of the repo can be found there.
  * `WoodFisher/main.py` is the main file to run pruning from.
  * `WoodFisher/transfer_checkpoint.ipynb` contains the code to transfer pre-trained checkpoint produced in our code to fit the WoodFisher pruning code.
  * `WoodFisher/checkpoints` is used to store the pre-trained models for the later pruning.
  * `WoodFisher/configs` contains yaml config files used for specifying training and pruning schedules. In our work, we only utilize the pruning schedules.
  * `WoodFisher/scripts` contains the all bash scripts for pruning to reproduce the results in the paper.
  * `WoodFisher/record_pruning` contains the code for visualizing the results after pruning.
  * `WoodFisher/lmc_source` contains edited code from our repo for applying re-normalization after pruning.

## Weights & Biases

We use the Weight & Biases (wandb) platform for logging results during training. To use wandb for the first time, you need to create an account and login. The `--wandb-mode` flag can be used to specify the mode of wandb. The default mode is `online`, which will log the results to the wandb server. If you want to run the code without logging to wandb, you can set `--wandb-mode` to `disabled`. If you want to log the results to wandb but do not want to create an account, you can set `--wandb-mode` to `offline`. In this case, the results will be logged to a local directory `wandb/` and you can upload the results to wandb later. For more details, please refer to the [wandb documentation](https://docs.wandb.ai/).

## Args description

### Training

We use a bash script to specify all training settings. The bash script is located in `training_scripts/`. All settings can be found in `source/utils/opts.py` with explanations. Here we only list some important args.

* `--project`: The wandb project name.
* `--run-name`: The wandb run name.
* `--dataset`: The dataset to use. We use `mnist`, `cifar10` and `cifar100` in our work.
* `--data-dir`: The dataset directory.
* `--model`: The model to use, including VGG and ResNet type of models.
  * Standard plain VGG models includ `cifar_vgg11`, `cifar_vgg13`, `cifar_vgg16`, and `cifar_vgg19`. VGG model with batch normalization is named with `_bn` suffix, e.g., `cifar_vgg11_bn`.
  * Standard ResNet models are named as `cifar_resnet[xx]`, e.g., `cifar_resnet20`. Plain/Fixup ResNet model is named with `plain_`/`fixup_` prefix, e.g., `plain_cifar_resnet20` and `fixup_cifar_resnet32`.
  * Models with layer normalization are named with `_ln` suffix, e.g., `cifar_vgg11_ln` and `cifar_resnet20_ln`.
  * Models without biases are named with `_nobias` suffix, e.g., `cifar_vgg11_nobias`.
  * Models with a larger width are named with `_[width_multipler]x` at the end, e.g., `cifar_vgg16_bn_4x`.
* `--diff-init`: Whether to use different initialization for the two models. If `True`, the two models are initialized with different random seeds.
* `--special-init`: Whether to use special initialization for models. Default is `None` and the models are initialized with the default Kaiming uniform initialization in PyTorch. If set to `vgg_init`, the Kaiming normal initialization is used.
* `--train-only`: Whether to only train the model without measuring the linear interpolation between the two models during training. If not, the linear interpolation is measured every `--lmc-freq` percent of the training.
* `--reset-bn`: Whether to reset BN statistics when measuring the linear interpolation during training.
* `--repair`: Whether to apply REPAIR/RESCALE when measuring the linear interpolation during training. Default is `None` and no re-normalizaiont is applied. If set to `repair`, REPAIR is applied. If set to `rescale`, RESCALE is applied.

### Pruning

We refer a complete description to the original repo: [WoodFisher](https://github.com/IST-DASLab/WoodFisher). Here we only list some important args in the pruning scripts located in `WoodFisher/scripts/`, which are important for reproducing the results in the paper.

* `MODULES`: The modules to prune.
* `ROOT_DIR`: The root directory to store the results.
* `DATA_DIR`: The dataset directory.
* `PRUNERS`: The pruners to use. Option: `woodfisherblock` `globalmagni` `magni` `diagfisher`
* `--num-samples`: The number of samples to use for applying re-normalization. Default is `None`.
* `--from_checkpoint_path`: The path to the pre-trained checkpoint.

## WoodFisher

The pruning results reported in the paper are conducted based on the framework in [WoodFisher](https://github.com/IST-DASLab/WoodFisher). Code is stored in `WoodFisher`. We manually edit some code in the original repo to force a one-shot pruning and remove some irrelevant feautres, especially for the `WoodFisher/policies/manager.py` file, while this can also be done by modifying the pruning settings in the scripts. The original file is retained in `WoodFisher/policies/manager_ori.py`. For applying re-normalization after pruning, we merged a modified version of our code with the repo, sotred in `WoodFisher/lmc_source/`. Several lines of code are also added to `WoodFisher/policies/manager.py`. This can be used as an example to merge our code with other pruning frameworks.

We'll release pre-trained checkpoints for re-producing the pruning results reported in the paper soon. These checkpoints were already transferred and hence there is no need to run the `WoodFisher/transfer_checkpoint.ipynb`.

### Setup

We replicate the setup of the original repo in the following.

First, clone the repository by running the following command:

```bash
git clone https://github.com/IST-DASLab/WoodFisher
```

After that, do not forget to run `source setup.sh`.

#### Tensorboard Support

First of all ensure that your `torch` package has version 1.1.0 or above. Then install the nightly release of `tensorboard`:

```bash
pip install tb-nightly
```

After that ensure that `future` package is installed or invoke installation process by typing the following command in terminal:

```bash
pip install future
```

## Reference

This codebase corresponds to the paper: *Rethink Model Re-Basin and the Linear Mode Connectivity*. If you use any of the code or provided models for your research, please consider citing the paper as

```bibtex
TODO
```
