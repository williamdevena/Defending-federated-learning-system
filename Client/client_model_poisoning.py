"""
This module contains the class that represents a malicious client that performs model poisononing attack.
"""
import random
import sys
import warnings
from collections import OrderedDict

import flwr as fl
import malicious_fun
import torch
import train_test
from net import Net

torch.cuda.empty_cache()

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define Flower client
class FlowerClientModelPoisoning(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, testloader):
        self.cid = cid
        self.attack_prob = random.uniform(0, 1)
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

    def get_cid(self):
        return self.cid

    def get_attack_prob(self):
        return self.attack_prob

    def get_parameters(self, config):
        if self.attack_prob > 0.5:
            parameters = [val.cpu().numpy() for _, val in self.net.state_dict().items()]
            noised_parameters = malicious_fun.add_noise_to_parameters(parameters)
            return noised_parameters
        else:
            return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_test.train(self.net, self.trainloader, epochs=1)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"cid": self.cid, "attack_prob": self.attack_prob}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = train_test.test(self.net, self.testloader)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy, "cid": self.cid, "attack_prob": self.attack_prob}

def main():
    # Load model and data (simple CNN, CIFAR-10)
    net = Net().to(DEVICE)
    trainloader, testloader = train_test.load_data()
    # Start Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClientModelPoisoning(
            cid=int(sys.argv[1]),
            net=net,
            trainloader=trainloader,
            testloader=testloader),
    )

if __name__=="__main__":
    main()