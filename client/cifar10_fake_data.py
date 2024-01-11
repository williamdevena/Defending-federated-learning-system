"""
This module contains the class CIFAR10WithFakeData that is used to implement the data poisoning attack.
"""

import random

import torch
from torchvision.datasets import CIFAR10

from typing import Tuple


class CIFAR10WithFakeData(CIFAR10):
    """
    A subclass of CIFAR10 dataset to simulate data poisoning attack by introducing fake data.

    This class is used to inject fake data into the CIFAR-10 dataset, allowing for testing of
    federated learning systems' resilience to data poisoning attacks.
    """
    def __init__(self, *args, fake_data_ratio: float = 0.1, **kwargs):
        """
        Initializes the CIFAR10WithFakeData dataset with the given fake data ratio.

        Args:
            *args: Variable length argument list for CIFAR10.
            fake_data_ratio (float): The proportion of fake data in the dataset.
            **kwargs: Arbitrary keyword arguments for CIFAR10.
        """
        super().__init__(*args, **kwargs)
        self.fake_data_ratio = fake_data_ratio

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Returns an item from the dataset, which is either a real CIFAR-10 data point or fake data.

        Args:
            index (int): The index of the item.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing an image and its label, which could be either real or fake.
        """
        if random.random() < self.fake_data_ratio:
            # Generate fake data
            fake_image = torch.rand(3, 32, 32)  # Random tensor with the same size as CIFAR-10 images
            fake_label = random.randint(0, 9)   # Random label from 0 to 9
            return fake_image, fake_label
        else:
            return super().__getitem__(index)