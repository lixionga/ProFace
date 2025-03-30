from math import exp
import torch
import torch.nn as nn
from models.rrdb_denselayer import ResidualSequential


class AffineBlock(nn.Module):
    def __init__(self, subnet_constructor=ResidualSequential, in_1=3, in_2=3,
                  password=True):
        super().__init__()
        self.split_len1 = 512
        self.split_len2 = 512
        self.clamp = 2
        self.password_channel = 16
        # ρ
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1)
        # ψ
        self.p = subnet_constructor(self.split_len2, self.split_len1)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def c(self, x, password):
        return torch.cat((x, password), 1)
    def forward(self, x, password=None, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))
        if not rev:

            t2 = self.f(x2)  # 16，3，64，
            s2 = self.p(x2)
            y1 = self.e(s2) * x1 + t2
            s1 = self.r(y1)
            t1 = self.y(y1)
            y2 = self.e(s1) * x2 + t1
        else:  # names of x and y are swapped!
            s1 = self.r(x1)
            t1 = self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            s2 = self.p(y2)
            y1 = (x1 - t2) / self.e(s2)
        return torch.cat((y1,y2),1)