import os
import numpy as np
import torch
from torchvision import datasets, transforms

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