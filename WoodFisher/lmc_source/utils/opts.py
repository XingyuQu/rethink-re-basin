import argparse


def save_arguments_to_file(parser, filename):
    with open(filename, 'w') as file:
        for action in parser._actions:
            arg_str = ', '.join(action.option_strings)
            default_str = str(action.default)
            file.write(f"{arg_str} (default: {default_str})\\\n")


def parse_args():
    parser = create_parser()
    args = parser.parse_args()
    return args


def create_parser():
    parser = argparse.ArgumentParser(description='Args Example')

    parser.add_argument('--project', type=str,
                        help='name of wandb project')
    parser.add_argument('--run-name', type=str,
                        help='name of wandb run')
    parser.add_argument('--run-notes', type=str,
                        help='notes of wandb run')
    parser.add_argument('--batch-size', type=int,
                        help='input batch size for training')
    parser.add_argument('--dataset', type=str,
                        help='dataset')
    parser.add_argument('--epochs', type=int,
                        help='number of epochs to train')
    parser.add_argument('--lmc-freq', type=float,
                        help='lmc test frequency '
                        '(e.g. 0.1 means 10% of epochs)')
    parser.add_argument('--lr', type=float,
                        help='learning rate')
    parser.add_argument('--model', type=str,
                        help='model architecture used in experiments')
    parser.add_argument('--momentum', type=float,
                        help='momentum')
    parser.add_argument('--n', type=int,
                        help='number of points to test lmc')
    parser.add_argument('--optimizer', type=str,
                        help='optimizer')
    parser.add_argument('--data-dir', type=str,
                        help='directory fo dataset')
    parser.add_argument('-p', '--print-freq', type=int,
                        help='print frequency')
    parser.add_argument('--seed', type=int, metavar='S',
                        help='random seed')
    parser.add_argument('--test-freq', type=int,
                        help='test frequency (e.g. 10 epochs)')
    parser.add_argument('--with-od', action='store_true', default=False,
                        help='use OD regularization (default False)')
    parser.add_argument('--sampler', type=str,
                        help='sampler (default: None)')
    parser.add_argument('--with-dp', action='store_true', default=False,
                        help='use dropout (default False)')
    parser.add_argument('--wd', type=float,
                        help='weight decay (default 0))')
    parser.add_argument('--reset-bn', action='store_true', default=False,
                        help='reset batch norm statistics before testing \
                            (default False)')
    # lr scheduler
    parser.add_argument('--scheduler', type=str,
                        help='scheduler (default: None)')
    parser.add_argument('--milestones', type=str,
                        help='milestones for lambda scheduler (default: None)')
    parser.add_argument('--warmup-iters', type=int,
                        help='warmup iterations for git rebasin scheduler \
                        (default: None)')
    parser.add_argument('--decay-iters', type=int,
                        help='decay iterations for git rebasin scheduler \
                        (default: None)')

    parser.add_argument('--wandb-mode', type=str,
                        help='whether to submit run to wanbd server \
                        (default: disabled)')
    parser.add_argument('--diff-init', action='store_true', default=False,
                        help='use different initialization for each model')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='save checkpoints of models (default: False)')
    parser.add_argument('--save-freq', type=int,
                        help='frequency of saving checkpoints of models \
                        (default: 10)')
    parser.add_argument('--ckpt-dir', type=str,
                        help='directory of checkpoints')
    parser.add_argument('--save-dir', type=str,
                        help='directory to save models')
    parser.add_argument('--special-init', type=str,
                        help='special initialization methodfor models')
    parser.add_argument('--subset', action='store_true', default=False,
                        help='use subset of data (default: False)')
    parser.add_argument('--device', type=str,
                        help='specify which device to use')
    parser.add_argument('--train-only', action='store_true', default=False,
                        help='only train two models, w/o lmc (default: False)')
    parser.add_argument('--init-model', action='store_true', default=False,
                        help='initialize model with pretrained model')
    parser.add_argument('--init-model-path-1', type=str,
                        help='path to pretrained model 1')
    parser.add_argument('--init-model-path-2', type=str,
                        help='path to pretrained model 2')
    parser.add_argument('--init-model-name-1', type=str,
                        help='name to pretrained model 1')
    parser.add_argument('--init-model-name-2', type=str,
                        help='name to pretrained model 2')
    parser.add_argument('--max-iter', type=int,
                        help='max number of iterations in weight matching')
    parser.add_argument('--repair',  action='store_true', default=False,
                        help='repair lmc')
    parser.add_argument('--mnist-mlp-init-ratio', type=float,
                        help='ratio * std in mnist mlp')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='use nesterov momentum in SGD')
    parser.add_argument('--no-random-aug', action='store_true', default=False,
                        help='disable random augmentation in dataset (cifar)')
    parser.add_argument('--save-lmc-path', type=str,
                        help='path to save lmc results')

    parser.add_argument('--otfusion', type=str,
                        help='otfusion method')

    parser.add_argument('--record-fim', action='store_true', default=False,)
    return parser
