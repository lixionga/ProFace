import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.normalization import SPADE

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import SPADEResBlk, ResBlk, UpBlock, ShuffleRes2Block

from options.train_options import TrainOptions
opt = TrainOptions().parse()


class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, output_last_feature=False):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        if output_last_feature:
            return h_relu4
        else:
            return h_relu4, h_relu3, h_relu2, h_relu1


vgg19=VGG19(requires_grad=False)
print(vgg19)


class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.opt.ngf = 128
        self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * self.opt.ngf, 3, padding=1)
        self.sh = self.opt.height // (2**5)  # fixed, 5 upsample layers
        self.sw = self.opt.width // (2**5)  # fixed, 5 upsample layers
        self.head = SPADEResBlk(16 * self.opt.ngf, 16 * self.opt.ngf, self.opt.semantic_nc)
        self.G_middle_0 = SPADEResBlk(16 * self.opt.ngf, 16 * self.opt.ngf, self.opt.semantic_nc)
        self.G_middle_1 = SPADEResBlk(16 * self.opt.ngf, 16 * self.opt.ngf, self.opt.semantic_nc)
        self.up_0 = SPADEResBlk(16 * self.opt.ngf, 8 * self.opt.ngf, self.opt.semantic_nc)
        if self.opt.multiscale_level == 4:
            self.up_1 = SPADEResBlk(8 * self.opt.ngf, 4 * self.opt.ngf, self.opt.semantic_nc // 2)
            self.up_2 = SPADEResBlk(4 * self.opt.ngf, 2 * self.opt.ngf, self.opt.semantic_nc // 4)
            self.up_3 = SPADEResBlk(2 * self.opt.ngf, 1 * self.opt.ngf, self.opt.semantic_nc // 8)
        elif self.opt.multiscale_level == 3:
            self.up_1 = SPADEResBlk(8 * self.opt.ngf, 4 * self.opt.ngf, self.opt.semantic_nc // 2)
            self.up_2 = SPADEResBlk(4 * self.opt.ngf, 2 * self.opt.ngf, self.opt.semantic_nc // 4)
            self.up_3 = SPADEResBlk(2 * self.opt.ngf, 1 * self.opt.ngf, self.opt.semantic_nc // 4)
        elif self.opt.multiscale_level == 2:
            self.up_1 = SPADEResBlk(8 * self.opt.ngf, 4 * self.opt.ngf, self.opt.semantic_nc // 2)
            self.up_2 = SPADEResBlk(4 * self.opt.ngf, 2 * self.opt.ngf, self.opt.semantic_nc // 2)
            self.up_3 = SPADEResBlk(2 * self.opt.ngf, 1 * self.opt.ngf, self.opt.semantic_nc // 2)
        elif self.opt.multiscale_level == 1:
            self.up_1 = SPADEResBlk(8 * self.opt.ngf, 4 * self.opt.ngf, self.opt.semantic_nc)
            self.up_2 = SPADEResBlk(4 * self.opt.ngf, 2 * self.opt.ngf, self.opt.semantic_nc)
            self.up_3 = SPADEResBlk(2 * self.opt.ngf, 1 * self.opt.ngf, self.opt.semantic_nc)
        self.conv_img = nn.Conv2d(self.opt.ngf, 3, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def up(x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, warped_features,a_features):
        # separate execute
        x = F.interpolate(warped_features[0], (self.sh, self.sw), mode='bilinear', align_corners=False)
        x = self.fc(x)
        x = self.head(x, 0.1*warped_features[0]+0.9*a_features[0])

        x = self.up(x)
        x = self.G_middle_0(x,0.1*warped_features[0]+0.9*a_features[0])
        x = self.G_middle_1(x,0.1*warped_features[0]+0.9*a_features[0])

        x = self.up(x)
        x = self.up_0(x, 0.1*warped_features[0]+0.9*a_features[0])

        if self.opt.multiscale_level == 4:
            x = self.up(x)
            x = self.up_1(x, 0.2*warped_features[1]+0.8*a_features[1])
            x = self.up(x)
            x = self.up_2(x, 0.4*warped_features[2]+0.6*a_features[2])
            x = self.up(x)
            x = self.up_3(x, warped_features[3])
        elif self.opt.multiscale_level == 3:
            x = self.up(x)
            x = self.up_1(x, warped_features[1])
            x = self.up(x)
            x = self.up_2(x, warped_features[2])
            x = self.up(x)
            x = self.up_3(x, warped_features[2])
        elif self.opt.multiscale_level == 2:
            x = self.up(x)
            x = self.up_1(x, warped_features[1])
            x = self.up(x)
            x = self.up_2(x, warped_features[1])
            x = self.up(x)
            x = self.up_3(x, warped_features[1])
        elif self.opt.multiscale_level == 1:
            x = self.up(x)
            x = self.up_1(x, warped_features[0])
            x = self.up(x)
            x = self.up_2(x, warped_features[0])
            x = self.up(x)
            x = self.up_3(x, warped_features[0])

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)
        return x

Decoder=SPADEGenerator(opt)
print(Decoder)
