"""
Dataset loading, normalisation and the train/val split.

One entry point matters: `build_dataloaders(config)` returns the (train, test)
loaders for whatever `dataset.name` and `training.*` say, resolving the download
root from `paths.dataset_path`. The per-dataset helpers below it
(load_mnist/load_fashion_mnist/load_cifar10) are what it dispatches to.

Dataset names are normalised once, here, by `normalize_dataset_name` ("cifar"
-> "cifar10" and so on). Everything downstream (checkpoint filenames, result
JSON metadata, the DATASET_SHAPES lookup in model_builder) keys off that
canonical spelling, so a config typo fails loudly rather than silently training
a second copy of a model under a different name.
"""
import copy
import os

import torch
from torch.utils.data import DataLoader, Subset
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

# Root under which torchvision stores its datasets, overridable per config with
# paths.dataset_path. hpc/prep_data.sh stages the downloads into the same root.
DEFAULT_DATA_ROOT = "data"

MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

NORMALIZATION = {
    "mnist": (MNIST_MEAN, MNIST_STD),
    "fashion_mnist": (MNIST_MEAN, MNIST_STD),
    "cifar10": (CIFAR10_MEAN, CIFAR10_STD),
}


def normalize_dataset_name(name):
    key = str(name).strip().lower()
    if key not in DATASET_ALIASES:
        raise ValueError(
            f"Unknown dataset: {name!r}. Supported: 'mnist', 'fashion_mnist', 'cifar10' "
            "(aliases: 'cifar', 'cifar-10', 'fashionmnist', 'fashion-mnist')."
        )
    return DATASET_ALIASES[key]


def resolve_data_root(config):
    """Dataset root directory (`paths.dataset_path`), defaulting to `data/`."""
    root = (config.get("paths", {}) or {}).get("dataset_path")
    return str(root) if root else DEFAULT_DATA_ROOT


def build_transform(dataset_name, augment=False):
    """
    Preprocessing pipeline for a dataset.

    `augment=True` adds the standard CIFAR-10 train-time augmentation (RandomCrop
    with padding 4 + horizontal flip). Evaluation, validation and Fisher
    computation must always use `augment=False` — they have to see clean images.
    """
    name = normalize_dataset_name(dataset_name)
    tfs = []
    if augment and name == "cifar10":
        tfs += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    tfs += [transforms.ToTensor(), transforms.Normalize(*NORMALIZATION[name])]
    return transforms.Compose(tfs)


def _make_loader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0),
    )


def load_mnist(batch_size=64, train=True, download=True, num_workers=4, pin_memory=True,
               data_root=DEFAULT_DATA_ROOT):
    dataset = datasets.MNIST(root=os.path.join(data_root, "MNIST"), train=train,
                             download=download, transform=build_transform("mnist"))
    return _make_loader(dataset, batch_size, train, num_workers, pin_memory)


def load_fashion_mnist(batch_size=64, train=True, download=True, num_workers=4, pin_memory=True,
                       data_root=DEFAULT_DATA_ROOT):
    dataset = datasets.FashionMNIST(root=os.path.join(data_root, "FashionMNIST"), train=train,
                                    download=download, transform=build_transform("fashion_mnist"))
    return _make_loader(dataset, batch_size, train, num_workers, pin_memory)


def load_cifar10(batch_size=64, train=True, download=True, num_workers=4, pin_memory=True,
                 augment=False, data_root=DEFAULT_DATA_ROOT):
    """
    CIFAR-10 loader. `augment=True` adds the standard train-time augmentation and
    should only be used for SGD training — Fisher computation and evaluation must
    see clean images.
    """
    dataset = datasets.CIFAR10(root=os.path.join(data_root, "CIFAR10"), train=train,
                               download=download,
                               transform=build_transform("cifar10", augment=(train and augment)))
    return _make_loader(dataset, batch_size, train, num_workers, pin_memory)


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

    common = dict(batch_size=batch_size, download=True, num_workers=num_workers,
                  pin_memory=pin_memory, data_root=resolve_data_root(config))

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


def split_train_val(config, train_loader, val_split, seed=None):
    """
    Hold out `val_split` of the training set for per-epoch validation.

    Returns `(train_loader, val_loader)`. A `val_split` of 0 (the default, and
    what the paper runs) returns the loader untouched and `None`, so the headline
    results train on the full split exactly as before.

    The validation subset is served WITHOUT augmentation even when the training
    subset has it: both Subsets index shallow copies of the same dataset object
    that differ only in `.transform`, so the image data is shared, not duplicated.
    The partition is drawn from `seed`, so it is reproducible per run.
    """
    if not val_split:
        return train_loader, None
    val_split = float(val_split)
    if not 0.0 < val_split < 1.0:
        raise ValueError(
            f"evaluation.validation_split must be in [0, 1), got {val_split}")

    dataset_name = normalize_dataset_name((config.get("dataset", {}) or {}).get("name", "mnist"))
    train_ds = train_loader.dataset
    val_ds = copy.copy(train_ds)
    val_ds.transform = build_transform(dataset_name, augment=False)

    n = len(train_ds)
    n_val = int(round(n * val_split))
    if not 0 < n_val < n:
        raise ValueError(
            f"evaluation.validation_split={val_split} leaves {n_val} of {n} samples "
            "for validation; choose a split that yields at least one sample in each part.")

    g = torch.Generator()
    g.manual_seed(0 if seed is None else int(seed))
    perm = torch.randperm(n, generator=g).tolist()

    kwargs = dict(batch_size=train_loader.batch_size, num_workers=train_loader.num_workers,
                  pin_memory=train_loader.pin_memory)
    return (_make_loader(Subset(train_ds, perm[n_val:]), shuffle=True, **kwargs),
            _make_loader(Subset(val_ds, perm[:n_val]), shuffle=False, **kwargs))
