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

torch.cuda.empty_cache()
# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# def train(net, trainloader, epochs):
#     """Train the model on the training set."""
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#     for _ in range(epochs):
#         for images, labels in tqdm(trainloader):
#             optimizer.zero_grad()
#             criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
#             optimizer.step()


# def test(net, testloader):
#     """Validate the model on the test set."""
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, loss = 0, 0.0
#     with torch.no_grad():
#         for images, labels in tqdm(testloader):
#             outputs = net(images.to(DEVICE))
#             labels = labels.to(DEVICE)
#             loss += criterion(outputs, labels).item()
#             correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
#     accuracy = correct / len(testloader.dataset)
#     return loss, accuracy


# def load_data():
#     """Load CIFAR-10 (training and test set)."""
#     trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     trainset = CIFAR10("./data", train=True, download=True, transform=trf)
#     testset = CIFAR10("./data", train=False, download=True, transform=trf)
#     return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = train_test.load_data()


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid

    def get_cid(self):
        return self.cid

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_test.train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {"cid": self.cid}

    def evaluate(self, parameters, config):
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