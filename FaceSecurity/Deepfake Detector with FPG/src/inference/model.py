from torch import nn
from torch.nn import functional as F
from models.efficientnet_pytorch import EfficientNet


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Detector(nn.Module):

    def __init__(self):
        super(Detector, self).__init__()
        self.net = EfficientNet.from_pretrained("efficientnet-b4", advprop=True, num_classes=2)
        
        self.irr_block = nn.Sequential(SeparableConv2d(1792, 512, 3, 1, 1, bias=False),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(inplace=True))
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1)
        
        self.c_attn = nn.Sequential(nn.Linear(512, 256, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(256, 1792, bias=False),
                                    nn.Sigmoid())
        
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.4)
        self._fc = nn.Linear(1792, 2)


    def forward(self, x, hidden=None, alpha=None):
        if hidden is None:
            x = self.net(x, hidden)
            b, c = x.shape[0], x.shape[1]
            
            irr_feat = self.irr_block(x)
            
            irr_feat = self.pool(irr_feat).flatten(start_dim=1)
            attn = self.c_attn(irr_feat).view(b, c, 1, 1)
            
            x = self._avg_pooling(x) * attn
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._fc(x)
            
            return x
        else:
            x, feat = self.net(x, hidden, alpha)
            b, c = x.shape[0], x.shape[1]
            
            irr_feat = self.irr_block(x)
            irr_feat = self.pool(irr_feat).flatten(start_dim=1)
            attn = self.c_attn(irr_feat).view(b, c, 1, 1)
            
            q = self.fc(irr_feat)
            
            x = self._avg_pooling(x) * attn
            
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._fc(x)
            
            return (x, q), feat