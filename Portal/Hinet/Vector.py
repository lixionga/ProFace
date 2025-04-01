import torch
from torch import nn
# import config as c

class vector_var(nn.Module):
    def __init__(self, size):
        super(vector_var, self).__init__()
        A = torch.rand(3, size, size, device='cpu')
        self.A = nn.Parameter(A)

    def forward(self):
        return self.A