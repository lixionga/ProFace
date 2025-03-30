import torch.nn as nn
from models.hinet import MLP
import torch
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model = MLP()
    # z是密钥
    # def forward(self, x, password, rev=False):
    def forward(self, x, rev=False):
        if not rev:
            # out = self.model(x, password)
            out = self.model(x)

        else:
            # out = self.model(x, password, rev=True)
            out = self.model(x, rev=True)

        return out


def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            # param.data = c.init_scale * torch.randn(param.data.shape).cuda()
            param.data = 0.01 * torch.randn(param.data.shape).cuda(1)  # 用的hi后修改为0
            if split[-2] == 'conv5':
                param.data.fill_(0.)



class KeyEmbedding(nn.Module):
    def __init__(self, input_dim=16, hidden_dims=[128, 256, 512], output_dim=512):
        super(KeyEmbedding, self).__init__()
        layers = []
        for i in range(len(hidden_dims)):
            in_dim = input_dim if i == 0 else hidden_dims[i - 1]
            out_dim = hidden_dims[i]
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())  # 激活函数
        layers.append(nn.Linear(hidden_dims[-1], output_dim))  # 输出层
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)