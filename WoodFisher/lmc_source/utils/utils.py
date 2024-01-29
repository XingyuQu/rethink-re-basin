from ..models import cifar_vgg, cifar_resnet, fixup_cifar_resnet, plain_cifar_resnet


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
