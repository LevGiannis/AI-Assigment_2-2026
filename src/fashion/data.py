import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True)

def get_fashionmnist_loaders(data_dir, batch_size, dev_ratio=0.1, quick=False, seed=42, return_indices=False):
    """
    Loads FashionMNIST and returns train/dev/test DataLoaders and sizes.
    If return_indices=True, also returns the indices used for train/dev split (for testing/debug).
    Args:
        data_dir: str
        batch_size: int
        dev_ratio: float
        quick: bool
        seed: int
        return_indices: bool, default False. If True, also returns (train_indices, dev_indices)
    Returns:
        train_loader, dev_loader, test_loader, n_train, n_dev, n_test
        If return_indices=True, also returns train_indices, dev_indices
    """
    set_seed(seed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset_full = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    testset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    if quick:
        train_indices_full = list(range(200))
        test_indices = list(range(100))
        trainset = Subset(trainset_full, train_indices_full)
        testset = Subset(testset, test_indices)
    else:
        train_indices_full = list(range(len(trainset_full)))
    n_dev = int(len(train_indices_full) * dev_ratio)
    n_train = len(train_indices_full) - n_dev
    # Deterministic split
    generator = torch.Generator().manual_seed(seed)
    lengths = [n_train, n_dev]
    trainset, devset = random_split(Subset(trainset_full, train_indices_full), lengths, generator=generator)
    # Extract indices for train/dev
    train_indices = [trainset.indices[i] for i in range(len(trainset))] if isinstance(trainset, Subset) else None
    dev_indices = [devset.indices[i] for i in range(len(devset))] if isinstance(devset, Subset) else None
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(devset, batch_size=batch_size)
    test_loader = DataLoader(testset, batch_size=batch_size)
    if return_indices:
        return train_loader, dev_loader, test_loader, len(trainset), len(devset), len(testset), train_indices, dev_indices
    else:
        return train_loader, dev_loader, test_loader, len(trainset), len(devset), len(testset)
