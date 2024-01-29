import torchvision
from torchvision import transforms
import torch


def sample_spherical(sample_num, sample_dim):
    data = torch.randn(sample_num, sample_dim)
    data /= torch.norm(data, dim=1, keepdim=True)
    return data


class CustomDataset(torch.utils.data.Dataset):
    """Some Information about Dataset"""
    def __init__(self, data, targets):
        super(CustomDataset, self).__init__()
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]


def load_data(path, dataset, load_trainset=True, download=True,
              no_random_aug=False):
    dataset = dataset.lower()
    if dataset.startswith("cifar"):  # CIFAR-10/100
        if not no_random_aug:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)),
        ])

        if dataset == "cifar10":
            if load_trainset:
                trainset = torchvision.datasets.CIFAR10(
                    root=path, train=True, download=download,
                    transform=transform_train)
            testset = torchvision.datasets.CIFAR10(
                root=path, train=False, download=download,
                transform=transform_test)

        elif dataset == "cifar100":
            if load_trainset:
                trainset = torchvision.datasets.CIFAR100(
                    root=path, train=True, download=download,
                    transform=transform_train)
            testset = torchvision.datasets.CIFAR100(
                root=path, train=False, download=download,
                transform=transform_test)
        else:
            raise NotImplementedError(f'{dataset} is not implemented.')
    elif dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,))])

        if load_trainset:
            trainset = torchvision.datasets.MNIST(
                path, train=True, download=download,
                transform=transform)
        testset = torchvision.datasets.MNIST(
            path, train=False, download=download,
            transform=transform)
    elif dataset == "fashion_mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,))])

        if load_trainset:
            trainset = torchvision.datasets.FashionMNIST(
                path, train=True, download=download,
                transform=transform)
        testset = torchvision.datasets.FashionMNIST(
            path, train=False, download=download,
            transform=transform)
    else:
        raise NotImplementedError(f'{dataset} is not implemented.')

    if not load_trainset:
        trainset = None

    return trainset, testset
