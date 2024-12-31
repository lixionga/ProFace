import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import SPADEResBlk, ResBlk, UpBlock, ShuffleRes2Block


class MuFFGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.opt.ngf = 64
        self.fc = nn.Conv2d(8 * self.opt.ngf, 16 * self.opt.ngf, 3, padding=1)
        self.sh = self.opt.height // (2**5)  # fixed, 5 upsample layers
        self.sw = self.opt.width // (2**5)  # fixed, 5 upsample layers
        self.up = UpBlock(16 * self.opt.ngf, 8 * self.opt.ngf)  # 32 
        if self.opt.no_multi:
            self.ff0 = UpBlock(8 * self.opt.ngf, 8 * self.opt.ngf)  # 64
            self.ff1 = UpBlock(8 * self.opt.ngf, 4 * self.opt.ngf)  # 128
            self.ff2 = UpBlock(4 * self.opt.ngf, 2 * self.opt.ngf)  # 256
            self.ff3 = UpBlock(2 * self.opt.ngf, 1 * self.opt.ngf)  # 512
        else:
            self.ff0 = ShuffleRes2Block(8 * self.opt.ngf, 4 * self.opt.ngf)  # 64
            self.ff1 = ShuffleRes2Block(4 * self.opt.ngf, 2 * self.opt.ngf)  # 128
            self.ff2 = ShuffleRes2Block(2 * self.opt.ngf, 1 * self.opt.ngf)  # 256
            self.ff3 = ShuffleRes2Block(1 * self.opt.ngf, 1 * self.opt.ngf)  # 512
        self.conv_img = nn.Conv2d(self.opt.ngf, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, warped_features):
        # separate execute
        x = F.interpolate(warped_features[0], (self.sh, self.sw), mode='bilinear', align_corners=False) # how can I forget this one?
        x = self.fc(x)  # 1024, 16, 16

        x = self.up(x)  # 512, 32, 32
        
        if self.opt.no_multi:
            x = self.ff0(x)  # 512, 64, 64 
            x = self.ff1(x)  # 256, 128, 128 
            x = self.ff2(x)  # 128, 256, 256
            x = self.ff3(x)  # 64, 512, 512
        else:
            x = self.ff0(x, warped_features[0])  # 256, 128, 128
            x = self.ff1(x, warped_features[1])  # 128, 256, 256 
            x = self.ff2(x, warped_features[2])  # 64, 512, 512
            x = self.ff3(x, warped_features[3])  # 64, 512, 512

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)
        return x


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
