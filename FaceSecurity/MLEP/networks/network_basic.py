
import torch
import torch.nn as nn
from collections import OrderedDict
from networks.resnet import conv1x1,BasicBlock
from einops.layers.torch import Rearrange

def sequential(*args):
    """
    Advanced nn.Sequential.
    Args:
        nn.Sequential, nn.Module
    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
    return sequential(*L)


class PatchEmbedding(nn.Module):
    def __init__(self, i_dim=64, n_layer=3):
        super(PatchEmbedding, self).__init__()
        self.inplanes = i_dim
        layers = []
        for i in range(n_layer):
            o_dim = i_dim * 2
            layers.append(self._make_layer(BasicBlock, o_dim, 2, stride=2))
            i_dim = o_dim
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.patch_to_embedding = nn.Sequential(*layers, avgpool)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, input):
        xs=[]
        for i in range(input.size(0)):
            x = self.patch_to_embedding(input[i, :]) # b,h,w,64  -> b,1,1,512
            xs.append(x.squeeze(-1).squeeze(-1))     # b,1,1,512 -> n,(b,512)
        x = torch.stack(xs).permute(1,0,2)           # n,(b,512) -> b,n,512
        return x


class ResTransformer(nn.Module):
    # def __init__(self, opt):
    def __init__(self):
        super(ResTransformer, self).__init__()

        # n_channel       = opt['network']['n_channel']
        # n_cbr           = opt['network']['n_cbr']
        # n_basicblock    = opt['network']['n_basicblock']
        # encoder_layers  = opt['network']['encoder_layers']
        # nhead           = opt['network']['nhead']
        # dim_feedforward = opt['network']['dim_feedforward']
        # dropout         = opt['network']['dropout']
        # crop_size       = opt['train']['cropsize']
        # patch_size      = opt['network']['patch_size']
        # padding_size    = opt['network']['padding_size']


        n_channel       = 64
        n_cbr           = 18
        n_basicblock    = 3
        encoder_layers  = 6
        nhead           = 8
        dim_feedforward = 2048
        dropout         = 0.1

        crop_size       = 256
        patch_size      = 32
        padding_size    = 64
        
        self.num_patches = (int((crop_size - patch_size) / padding_size) + 1) ** 2

        # Network-1 Denoiser
        D_head = conv(3 , n_channel, mode='CR')                                 # b,h,w,3  -> b,h,w,nc
        D_body = [conv(n_channel, n_channel, mode='CBR') for _ in range(n_cbr)] # b,h,w,nc -> b,h,w,nc
        self.denoiser     = sequential(D_head, *D_body)
        self.return_image = conv(n_channel, 3,  mode='C')                       # b,h,w,nc -> b,h,w,3

        # Network-2 FeatureMap
        self.patches_to_embedding = PatchEmbedding(i_dim=n_channel, n_layer=n_basicblock) # b,h,w,nc -> b,n,nc*(nl**2)
        transformer_dim = n_channel * (2 ** n_basicblock)

        # Network-3 Results
        self.ReT = nn.Transformer(
            d_model = transformer_dim,
            num_encoder_layers = encoder_layers,
            num_decoder_layers = 0,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
        )
        self.fc =  sequential(
            Rearrange('b n f -> b (n f)'),
            nn.Linear(self.num_patches*transformer_dim, 1),
        )

    def forward(self, patches_L, show_networks=False, return_features=False):

        if show_networks:
            patches_E = []
            n_channel = 64
            batches = patches_L.unfold(2, n_channel, n_channel).unfold(3, n_channel, n_channel)
            for i in range(batches.size(2)):
                for j in range(batches.size(3)):
                    patch = batches[:, :, i, j, :] # [1, 3, 64, 64]
                    patches_E.append(patch)
            patches_L = patches_E

        patches_E, xs = [], []
        for patch_L in patches_L:
            x = self.denoiser(patch_L)
            xs.append(x)
            x_e = patch_L-self.return_image(x)
            patches_E.append(x_e)
        x = torch.stack(xs)
        x = self.patches_to_embedding(x)
        x = self.ReT(x, tgt=x)

        if return_features:
            return {'E': patches_E,'features': x}
        else:
            x = self.fc(x.squeeze(2))
            return {'E': patches_E,'label': x}


