import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm

input_dims = (512,)
cond_dims = (512,)
def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 256), nn.LeakyReLU(negative_slope=0.01, inplace=False),
                         nn.Linear(256, 128), nn.LeakyReLU(negative_slope=0.01, inplace=False),
                         nn.Linear(128, 128), nn.LeakyReLU(negative_slope=0.01, inplace=False),
                         nn.Linear(128, 256), nn.LeakyReLU(negative_slope=0.01, inplace=False),
                         nn.Linear(256, dims_out))

def bulid_models():
    G_net = Ff.SequenceINN(*input_dims)
    for k in range(4):
        G_net.append(Fm.AllInOneBlock, cond=0, cond_shape=cond_dims, subnet_constructor=subnet_fc)
    for m in G_net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
    return G_net