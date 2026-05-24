import torch
import torch.nn as nn


class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20).cuda()
        self.fc2 = nn.Linear(20, 10).cuda()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def simple_network_demo():
    network = SimpleNetwork()
    x = torch.randn(10, 10).cuda()
    y = network(x)
    expect = torch.randn(10, 10).cuda()
    loss = nn.functional.mse_loss(y, expect, reduction="sum")
    loss.backward()
    print(network.fc1.weight.grad)
    print(network.fc2.weight.grad)


if __name__ == "__main__":
    simple_network_demo()
