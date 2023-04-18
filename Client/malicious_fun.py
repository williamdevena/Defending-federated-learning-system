"""
This module contains functions used to implement attacks.
"""

import numpy as np
from cifar10_fake_data import CIFAR10WithFakeData
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


def add_noise_to_parameters(parameters, noise_factor=0.1):
    noised_parameters = []
    for param in parameters:
        noise = np.random.normal(0, noise_factor, param.shape)
        noised_param = param + noise
        noised_parameters.append(noised_param)
    return noised_parameters


# Update the data loading function
def load_noised_data():
    """Load CIFAR-10 (training and test set) with added fake data."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10WithFakeData("./data", train=True, download=True, transform=trf, fake_data_ratio=0.5)
    testset = CIFAR10WithFakeData("./data", train=False, download=True, transform=trf, fake_data_ratio=0.5)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)