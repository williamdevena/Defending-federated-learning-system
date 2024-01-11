"""
This module contains functions used to implement attacks.
"""

import numpy as np
from cifar10_fake_data import CIFAR10WithFakeData
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from typing import List, Tuple


def add_noise_to_parameters(parameters: List[np.ndarray],
                            noise_factor: float = 0.1) -> List[np.ndarray]:
    """
    Adds noise to the given parameters to simulate a model poisoning attack.

    Args:
        parameters (List[np.ndarray]): The original parameters of the model.
        noise_factor (float): The factor determining the amount of noise to add.

    Returns:
        List[np.ndarray]: The noised parameters.
    """
    noised_parameters = []
    for param in parameters:
        noise = np.random.normal(0, noise_factor, param.shape)
        noised_param = param + noise
        noised_parameters.append(noised_param)
    return noised_parameters


# Update the data loading function
def load_noised_data() -> Tuple[DataLoader, DataLoader]:
    """
    Loads the CIFAR-10 dataset with added fake data (CIFAR10WithFakeData)
    for training and testing.

    Returns:
        Tuple[DataLoader, DataLoader]: DataLoaders for the noised training and test sets.
    """
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10WithFakeData("./data", train=True, download=True, transform=trf, fake_data_ratio=0.5)
    testset = CIFAR10WithFakeData("./data", train=False, download=True, transform=trf, fake_data_ratio=0.5)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)