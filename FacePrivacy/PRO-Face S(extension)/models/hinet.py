import torch.nn as nn
from models.invblock import AffineBlock
class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.inv1 = AffineBlock()
        self.inv2 = AffineBlock()
        self.inv3 = AffineBlock()
        self.inv4 = AffineBlock()


    def forward(self, x, password = None, rev=False):

        if not rev:
            out = self.inv1(x, password)
            out = self.inv2(out, password)
            out = self.inv3(out, password)
            out = self.inv4(out, password)
        else:
            out = self.inv4(x, password, rev=True)
            out = self.inv3(out, password, rev=True)
            out = self.inv2(out, password, rev=True)
            out = self.inv1(out, password, rev=True)

        return out