"""
This module contains the class CIFAR10WithFakeData that is used to implement the data poisoning attack.
"""

import random

import torch
from torchvision.datasets import CIFAR10


class CIFAR10WithFakeData(CIFAR10):
    def __init__(self, *args, fake_data_ratio=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.fake_data_ratio = fake_data_ratio

    def __getitem__(self, index: int):
        if random.random() < self.fake_data_ratio:
            # Generate fake data
            fake_image = torch.rand(3, 32, 32)  # Random tensor with the same size as CIFAR-10 images
            fake_label = random.randint(0, 9)   # Random label from 0 to 9
            return fake_image, fake_label
        else:
            return super().__getitem__(index)