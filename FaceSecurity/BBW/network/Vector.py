import torch
from torch import nn
import FaceSecurity.BBW.config.cfg as c


class vector_var(nn.Module):
    def __init__(self, size):
        super(vector_var, self).__init__()
        A = torch.rand(c.set_channels, size, size, device='cpu')
        self.A = nn.Parameter(A)

    def forward(self):
        return self.A
