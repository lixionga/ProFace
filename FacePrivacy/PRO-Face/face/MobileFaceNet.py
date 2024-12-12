import torch.nn as nn
import torch
import math
# from apex import amp
# from Config import args

Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and in_channels == out_channels

        self.conv = nn.Sequential(
            #1*1 conv
            nn.Conv2d(in_channels, in_channels * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels * expansion),
            nn.PReLU(in_channels * expansion),

            #3*3 depth wise conv
            nn.Conv2d(in_channels * expansion, in_channels * expansion, 3, stride, 1, groups=in_channels * expansion, bias=False),
            nn.BatchNorm2d(in_channels * expansion),
            nn.PReLU(in_channels * expansion),

            #1*1 conv linear
            nn.Conv2d(in_channels * expansion, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear

        # conv, bn, prelu
        if dw:
            self.conv = nn.Conv2d(in_channels, out_channels, k, s, p, groups = in_channels, bias = False)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, k, s, p, bias = False)
        self.bn = nn.BatchNorm2d(out_channels)
        if not linear:
            self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)

class MobileFacenet(nn.Module):
    def __init__(self, bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(MobileFacenet, self).__init__()

        self.conv3 = ConvBlock(3, 64, 3, 2, 1)

        self.dw_conv3 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.in_channels = 64
        bottleneck = Bottleneck
        self.bottlenecks = self._make_layer(bottleneck, bottleneck_setting)

        self.conv1 = ConvBlock(128, 512, 1, 1, 0)

        self.linear_GDConv7 = ConvBlock(512, 512, 7, 1, 0, dw=True, linear=True)

        self.linear_conv1 = ConvBlock(512, 128, 1, 1, 0, linear=True)

        # parameter init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # kaiming_normal
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # # prevent overflow errors
        # if args.use_amp == True:
        #     amp.register_float_function(torch, 'sigmoid')
        #     amp.register_float_function(torch, 'softmax')

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.in_channels, c, s, t))
                else:
                    layers.append(block(self.in_channels, c, 1, t))
                self.in_channels = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv3(x)
        x = self.dw_conv3(x)
        x = self.bottlenecks(x)
        x = self.conv1(x)
        x = self.linear_GDConv7(x)
        x = self.linear_conv1(x)
        x = x.view(x.shape[0], -1)
        return x

if __name__ == "__main__":
    input = torch.Tensor(256, 3, 112, 112)
    net = MobileFacenet()
    x = net(input)
    print(input.shape)
    print(x.shape)