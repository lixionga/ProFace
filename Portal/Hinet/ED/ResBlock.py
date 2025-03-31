import torch
import torch.nn as nn
import torch.nn.functional as F


class SEAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(SEAttention, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels // reduction, out_channels=out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.se(x) * x
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels // reduction, out_channels=out_channels, kernel_size=1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAMAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(CBAMAttention, self).__init__()
        self.ca = ChannelAttention(in_channels=in_channels, out_channels=out_channels, reduction=reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(CoordAttention, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.InstanceNorm2d(temp_c)
        self.act1 = h_swish()  # nn.SiLU() # nn.Hardswish() # nn.SiLU()

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        return short * out_w * out_h


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction, stride, attention=None):
        super(BasicBlock, self).__init__()

        self.change = None
        if (in_channels != out_channels or stride != 1):
            self.change = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
                          stride=stride, bias=False),
                nn.InstanceNorm2d(out_channels)
            )

        self.left = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1,
                      stride=stride, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels)
        )

        if attention == 'se':
            print('SEAttention')
            self.attention = SEAttention(in_channels=out_channels, out_channels=out_channels, reduction=reduction)
        elif attention == 'cbam':
            print('CBAMAttention')
            self.attention = CBAMAttention(in_channels=out_channels, out_channels=out_channels, reduction=reduction)
        elif attention == 'coord':
            print('CoordAttention')
            self.attention = CoordAttention(in_channels=out_channels, out_channels=out_channels, reduction=reduction)
        else:
            print('None Attention')
            self.attention = nn.Identity()

    def forward(self, x):
        identity = x
        x = self.left(x)
        x = self.attention(x)

        if self.change is not None:
            identity = self.change(identity)

        x += identity
        x = F.relu(x)
        return x


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction, stride, attention=None):
        super(BottleneckBlock, self).__init__()

        self.change = None
        if (in_channels != out_channels or stride != 1):
            self.change = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
                          stride=stride, bias=False),
                nn.InstanceNorm2d(out_channels)
            )

        self.left = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                      stride=stride, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels)
        )

        if attention == 'se':
            print('SEAttention')
            self.attention = SEAttention(in_channels=out_channels, out_channels=out_channels, reduction=reduction)
        elif attention == 'cbam':
            print('CBAMAttention')
            self.attention = CBAMAttention(in_channels=out_channels, out_channels=out_channels, reduction=reduction)
        elif attention == 'coord':
            print('CoordAttention')
            self.attention = CoordAttention(in_channels=out_channels, out_channels=out_channels, reduction=reduction)
        else:
            print('None Attention')
            self.attention = nn.Identity()

    def forward(self, x):
        identity = x
        x = self.left(x)
        x = self.attention(x)

        if self.change is not None:
            identity = self.change(identity)

        x += identity
        x = F.relu(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, blocks=1, block_type="BottleneckBlock", reduction=8, stride=1,
                 attention=None):
        super(ResBlock, self).__init__()

        layers = [
            eval(block_type)(in_channels, out_channels, reduction, stride, attention=attention)] if blocks != 0 else []
        for _ in range(blocks - 1):
            layer = eval(block_type)(out_channels, out_channels, reduction, 1, attention=attention)
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
