from math import exp
import torch
import torch.nn as nn
import config.config as c
from rrdb_denselayer import ResidualDenseBlock_out


class INV_block(nn.Module):
    def __init__(self, subnet_constructor=ResidualDenseBlock_out, clamp=c.clamp, harr=True,
                 conditional=True, in_1=c.channels_in, in_2=c.channels_in):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 4
            self.split_len2 = in_2 * 4
        self.clamp = clamp
        password_channel = 4 if conditional else 0
        # ρ
        self.r = subnet_constructor(self.split_len1 + password_channel, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1 + password_channel, self.split_len2)
        # φ
        self.f = subnet_constructor(self.split_len2 + password_channel, self.split_len1)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def c(self, x, password):
        ''' Concate input x with a password channel wise '''
        return torch.cat((x, password), 1)

    def forward(self, x, password=None, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            t2 = self.f(self.c(x2, password)) # self.f(x2)
            y1 = x1 + t2
            # s1, t1 = self.r(y1), self.y(y1)
            s1 = self.r(self.c(y1, password)) # self.r(y1)
            t1 = self.y(self.c(y1, password)) # self.y(y1)
            y2 = self.e(s1) * x2 + t1

        else:
            # s1, t1 = self.r(x1), self.y(x1)
            s1 = self.r(self.c(x1, password)) # self.r(x1)
            t1 = self.y(self.c(x1, password)) # self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(self.c(y2, password)) # self.f(y2)
            y1 = (x1 - t2)

        return torch.cat((y1, y2), 1)


class INV_block_affine(nn.Module):
    def __init__(self, subnet_constructor=ResidualDenseBlock_out, clamp=c.clamp, harr=True, in_1=3, in_2=3,
                 imp_map=False, password=True):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 4
            self.split_len2 = in_2 * 4
        self.clamp = clamp
        if imp_map:
            self.imp = 12
        else:
            self.imp = 0

        self.password_channel = 4 if password else 0
        if c.ADJ_UTILITY:
            self.password_channel += 1

        # ρ
        self.r = subnet_constructor(self.split_len1 + self.imp + self.password_channel, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1 + self.imp + self.password_channel, self.split_len2)
        # φ
        self.f = subnet_constructor(self.split_len2 + self.password_channel, self.split_len1 + self.imp)
        # ψ
        self.p = subnet_constructor(self.split_len2 + self.password_channel, self.split_len1 + self.imp)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def c(self, x, password):
        ''' Concate input x with a password channel wise '''
        return torch.cat((x, password), 1)

    def forward(self, x, password=None, rev=False):

        x1, x2 = (x.narrow(1, 0, self.split_len1 + self.imp),
                  x.narrow(1, self.split_len1 + self.imp, self.split_len2))

        if not rev:

            # t2 = self.f(x2)
            t2 = self.f(self.c(x2, password))
            # s2 = self.p(x2)
            s2 = self.p(self.c(x2, password))
            y1 = self.e(s2) * x1 + t2
            # s1, t1 = self.r(y1), self.y(y1)
            s1 = self.r(self.c(y1, password))
            t1 = self.y(self.c(y1, password))
            y2 = self.e(s1) * x2 + t1

        else:  # names of x and y are swapped!

            # s1, t1 = self.r(x1), self.y(x1)
            s1 = self.r(self.c(x1, password))
            t1 = self.y(self.c(x1, password))
            y2 = (x2 - t1) / self.e(s1)
            # t2 = self.f(y2)
            t2 = self.f(self.c(y2, password))
            # s2 = self.p(y2)
            s2 = self.p(self.c(y2, password))
            y1 = (x1 - t2) / self.e(s2)

        return torch.cat((y1, y2), 1)

