"""
This module contains the class that represents a malicious client that performs data poisononing attack.
"""

import random
import sys
import warnings
from collections import OrderedDict
import numpy as np

import flwr as fl
import malicious_fun
import torch
import train_test
from net import Net
from typing import List, Tuple, Dict, Union

torch.cuda.empty_cache()

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define Flower client
class FlowerClientDataPoisoning(fl.client.NumPyClient):
    """
    A Flower client representing a malicious client in a federated learning system.

    This client performs data poisoning attacks, either using genuine data or noised data
    based on a randomly determined attack probability.
    """
    def __init__(self,
                 cid: int,
                 net: Net,
                 trainloader: torch.utils.data.DataLoader,
                 testloader: torch.utils.data.DataLoader,
                 noised_trainloader: torch.utils.data.DataLoader,
                 noised_testloader: torch.utils.data.DataLoader):
        self.cid = cid
        self.attack_prob = random.uniform(0, 1)
        #self.attack_prob = 0.99
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.noised_trainloader = noised_trainloader
        self.noised_testloader = noised_testloader


    def get_cid(self) -> int:
        """
        Returns the client ID.

        Returns:
            int: Client ID.
        """
        return self.cid


    def get_attack_prob(self) -> float:
        """
        Returns the attack probability.

        Returns:
            float: Attack probability.
        """
        return self.attack_prob


    def get_parameters(self) -> List[np.ndarray]:
        """
        Retrieves the parameters of the neural network model as a list of NumPy arrays.

        Returns:
            List[np.ndarray]: A list of NumPy arrays representing the model parameters.
        """
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]


    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Sets the parameters of the neural network model.

        Args:
            parameters (List[np.ndarray]): A list of NumPy arrays representing the model parameters to be set.
        """
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)


    def fit(self, parameters: List[np.ndarray]) -> Tuple[List[np.ndarray], int, Dict[str, Union[int, float]]]:
        """
        Trains the model on either genuine or noised data based on the attack probability.

        Args:
            parameters (List[np.ndarray]): Model parameters to be used for training.

        Returns:
            Tuple[List[np.ndarray], int, Dict[str, Union[int, float]]]: Updated model parameters, dataset length, and client information.
        """
        self.set_parameters(parameters)
        if self.attack_prob < 0.5:
            train_test.train(self.net, self.trainloader, epochs=1)
            length = len(self.trainloader.dataset)
        else:
            train_test.train(self.net, self.noised_trainloader, epochs=1)
            length = len(self.noised_trainloader.dataset)
        return self.get_parameters(config={}), length, {"cid": self.cid, "attack_prob": self.attack_prob}


    def evaluate(self, parameters: List[np.ndarray]) -> Tuple[float, int, Dict[str, Union[float, int]]]:
        """
        Evaluates the model using the test data loader.

        Args:
            parameters (List[np.ndarray]): Model parameters to be used for evaluation.
            config: Evaluation configuration.

        Returns:
            Tuple[float, int, Dict[str, Union[float, int]]]: Loss, dataset length, and evaluation metrics.
        """
        self.set_parameters(parameters)
        loss, accuracy = train_test.test(self.net, self.testloader)
        length = len(self.testloader.dataset)
        return loss, length, {"accuracy": accuracy, "cid": self.cid, "attack_prob": self.attack_prob}




def main():
    # Load model and data (simple CNN, CIFAR-10)
    net = Net().to(DEVICE)
    trainloader, testloader = train_test.load_data()
    noised_trainloader, noised_testloader = malicious_fun.load_noised_data()

    # Start Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClientDataPoisoning(
            cid=int(sys.argv[1]),
            net=net,
            trainloader=trainloader,
            testloader=testloader,
            noised_trainloader=noised_trainloader,
            noised_testloader=noised_testloader),
    )




if __name__=="__main__":
    main()