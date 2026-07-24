import os
import numpy as np
import torch
from torchvision import datasets, transforms


# Accepted spellings -> canonical dataset name used everywhere downstream
# (checkpoint filenames, result JSON metadata, DATASET_SHAPES lookups).
DATASET_ALIASES = {
    "mnist": "mnist",
    "fashion_mnist": "fashion_mnist",
    "fashionmnist": "fashion_mnist",
    "fashion-mnist": "fashion_mnist",
    "cifar10": "cifar10",
    "cifar": "cifar10",
    "cifar-10": "cifar10",
}


def normalize_dataset_name(name):
    key = str(name).strip().lower()
    if key not in DATASET_ALIASES:
        raise ValueError(
            f"Unknown dataset: {name!r}. Supported: 'mnist', 'fashion_mnist', 'cifar10' "
            "(aliases: 'cifar', 'cifar-10', 'fashionmnist', 'fashion-mnist')."
        )
    return DATASET_ALIASES[key]


def load_mnist(batch_size=64, train=True, download=True, num_workers=4, pin_memory=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root=os.path.join('data', 'MNIST'), train=train, download=download, transform=transform)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=train,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0),
    )

def load_fashion_mnist(batch_size=64, train=True, download=True, num_workers=4, pin_memory=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.FashionMNIST(root=os.path.join('data', 'FashionMNIST'), train=train, download=download, transform=transform)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=train,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0),
    )


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

def load_cifar10(batch_size=64, train=True, download=True, num_workers=4, pin_memory=True, augment=False):
    """
    CIFAR-10 loader. `augment=True` adds the standard train-time augmentation
    (RandomCrop with padding 4 + horizontal flip) and should only be used for
    SGD training — Fisher computation and evaluation must see clean images.
    """
    tfs = []
    if train and augment:
        tfs += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    tfs += [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    transform = transforms.Compose(tfs)
    dataset = datasets.CIFAR10(root=os.path.join('data', 'CIFAR10'), train=train, download=download, transform=transform)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=train,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0),
    )


def build_dataloaders(config, augment=False):
    """
    Shared dataset dispatch (previously duplicated across train_model.py,
    run_pruning.py and subset_experiment.py).

    Returns:
        (dataset_name, train_loader, test_loader)
    """
    dataset_name = normalize_dataset_name((config.get("dataset", {}) or {}).get("name", "mnist"))
    batch_size = config["training"]["batch_size"]
    num_workers = int(config.get("training", {}).get("num_workers", 4))
    pin_memory = bool(config.get("training", {}).get("pin_memory", True))

    common = dict(batch_size=batch_size, download=True, num_workers=num_workers, pin_memory=pin_memory)

    if dataset_name == "mnist":
        train_loader = load_mnist(train=True, **common)
        test_loader = load_mnist(train=False, **common)
    elif dataset_name == "fashion_mnist":
        train_loader = load_fashion_mnist(train=True, **common)
        test_loader = load_fashion_mnist(train=False, **common)
    elif dataset_name == "cifar10":
        train_loader = load_cifar10(train=True, augment=augment, **common)
        test_loader = load_cifar10(train=False, **common)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset_name, train_loader, test_loader
