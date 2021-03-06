"""
Data loader utils
"""
from typing import Any, Optional

import torch
import torchvision.datasets
import torch.utils.data as data_utils
import torchvision.transforms as transforms

DATASET = ['CIFAR10', 'CIFAR100', 'IMAGENET64']


def create_dataloader(
    dataset: str,
    data_dir: str,
    split: Optional[str] = 'train',
    batch_size: Optional[int] = 32,
    shuffle: Optional[bool] = True
) -> Any:
    is_train = True if split == 'train' else False

    if dataset == 'CIFAR10':
        print("Transforming dataset...")
        if is_train:
            transform_ops = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_ops = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        dataset = torchvision.datasets.CIFAR10(
            data_dir, train=is_train, transform=transform_ops
        )
    elif dataset == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100(
            data_dir, train=is_train, transform=transforms.ToTensor()
        )

    loader = data_utils.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=8, pin_memory=True
    )
    return loader
