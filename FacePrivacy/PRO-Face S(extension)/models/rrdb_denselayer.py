import torch
import torch.nn as nn


class ResidualSequential(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualSequential, self).__init__()
        self.model = nn.Sequential(nn.Linear(input, 256), nn.LeakyReLU(negative_slope=0.01, inplace=False),
                      nn.Linear(256, 128), nn.LeakyReLU(negative_slope=0.01, inplace=False),
                      nn.Linear(128, 128), nn.LeakyReLU(negative_slope=0.01, inplace=False),
                      nn.Linear(128, 256), nn.LeakyReLU(negative_slope=0.01, inplace=False),
                      nn.Linear(256, output))

    def forward(self, x):
        return self.model(x)
