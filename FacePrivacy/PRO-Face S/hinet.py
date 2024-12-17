# from face_detection import *
from torch import nn
from invblock import INV_block, INV_block_affine


class Hinet(nn.Module):

    def __init__(self, n_blocks=6):
        super(Hinet, self).__init__()
        self.inv_blocks = nn.ModuleList([INV_block_affine() for _ in range(n_blocks)])

    def forward(self, x, password, rev=False):
        if not rev:
            for inv_block in self.inv_blocks:
                x = inv_block(x, password)
        else:
            for inv_block in reversed(self.inv_blocks):
                x = inv_block(x, password, rev=True)
        return x

