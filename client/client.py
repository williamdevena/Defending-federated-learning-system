"""
This module contains the class that represents a benevolent client.
"""

import sys
import warnings
from collections import OrderedDict

import flwr as fl
import torch
import train_test
from net import Net
from typing import List, Tuple, Dict, Union
import numpy as np

torch.cuda.empty_cache()

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = train_test.load_data()

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    """
    A Flower client representing a benevolent client in a federated learning system.

    This client participates in the federated learning process without any malicious intent,
    contributing to the training and evaluation of the shared model.
    """
    def __init__(self, cid: int) -> None:
        self.cid = cid


    def get_cid(self) -> int:
        """
        Returns the client ID.

        Returns:
            int: Client ID.
        """
        return self.cid


    def get_parameters(self) -> List[np.ndarray]:
        """
        Retrieves the parameters of the neural network model as a list of NumPy arrays.

        Args:
            config (Dict): Configuration for getting parameters.

        Returns:
            List[np.ndarray]: A list of NumPy arrays representing the model parameters.
        """
        return [val.cpu().numpy() for _, val in net.state_dict().items()]


    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Sets the parameters of the neural network model.

        Args:
            parameters (List[np.ndarray]): A list of NumPy arrays representing the model parameters to be set.
        """
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)


    def fit(self, parameters: List[np.ndarray]) -> Tuple[List[np.ndarray], int, Dict[str, int]]:
        """
        Trains the model on the training data loader.

        Args:
            parameters (List[np.ndarray]): Model parameters to be used for training.
            config (Dict): Training configuration.

        Returns:
            Tuple[List[np.ndarray], int, Dict[str, int]]: Updated model parameters, dataset length, and client information.
        """
        self.set_parameters(parameters)
        train_test.train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {"cid": self.cid}


    def evaluate(self, parameters: List[np.ndarray]) -> Tuple[float, int, Dict[str, Union[float, int]]]:
        """
        Evaluates the model using the test data loader.

        Args:
            parameters (List[np.ndarray]): Model parameters to be used for evaluation.
            config (Dict): Evaluation configuration.

        Returns:
            Tuple[float, int, Dict[str, Union[float, int]]]: Loss, dataset length, and evaluation metrics.
        """
        self.set_parameters(parameters)
        loss, accuracy = train_test.test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy, "cid": self.cid}


def main():
    # Start Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(cid=int(sys.argv[1])),
    )


if __name__=="__main__":
    main()