"""
Data loader utils
"""
from typing import Any, Optional

import torchvision.datasets
import torch.utils.data as data_utils
import torchvision.transforms as transforms


def create_dataloader(
    dataset: str,
    data_dir: str,
    split: Optional[str] = 'train',
    batch_size: Optional[int] = 32,
    shuffle: Optional[bool] = True
) -> Any:
    is_train = True if split == 'train' else False

    if dataset == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(
            data_dir, train=is_train, transform=transforms.ToTensor()
        )
    elif dataset == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100(
            data_dir, train=is_train, transform=transforms.ToTensor()
        )

    loader = data_utils.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=16, pin_memory=True
    )
    return loader
