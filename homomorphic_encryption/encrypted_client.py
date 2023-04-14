import inspect
import warnings
from collections import OrderedDict
from io import BytesIO
from pprint import pprint

import encryption
import flower_overwrite
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common.typing import Parameters
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        print("---------------------------")
        print("---------------------------")
        print("---------------------------")
        print("get parameters")
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        print('callers name:', [(fun[1], fun[3]) for fun in calframe])
        #print(calframe)
        print("---------------------------")
        print("---------------------------")
        print("---------------------------")
        # print([val.cpu().numpy().shape for _, val in net.state_dict().items()])

        list_enc_params = []
        for _, val in net.state_dict().items():
            np_param = val.cpu().numpy()
            orig_shape = np_param.shape
            np_param_flatten = np_param.flatten()
            # print(orig_shape)
            # print(np_param_flatten.shape)
            enc_param = encryption.encrypt_parameters(param=np_param.flatten())
            # print(np.array(enc_param.decrypt()).reshape(orig_shape).shape)
            # print("-----------")
            list_enc_params.append(enc_param)

        # print(list_enc_params)
        return list_enc_params
        #return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print("")
        print("")
        print("fit")
        print("")
        print("")
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}






def main():
  #fl.common.ndarrays_to_parameters = ndarrays_to_parameters_new
  # fl.common.ndarray_to_bytes = ndarray_to_bytes_new
  #fl.common.ndarray_to_bytes = ndarray_to_bytes_new
  fl.client.app.ndarrays_to_parameters = flower_overwrite.ndarrays_to_parameters_new
  fl.client.app.parameters_to_ndarrays = flower_overwrite.parameters_to_ndarrays_new
  fl.client.app._get_parameters = flower_overwrite.get_parameters2_new
  fl.common.serde.parameters_to_proto = flower_overwrite.parameters_to_proto_new

  # Start Flower client
  fl.client.start_numpy_client(
      server_address="127.0.0.1:8080",
      client=FlowerClient(),
  )

if __name__=="__main__":
  main()
