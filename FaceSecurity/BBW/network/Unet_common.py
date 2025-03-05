import math

import torch
import torch.nn as nn

from modules import module_util as mutil
import functools
import config as c
device = torch.device(c.device if torch.cuda.is_available() else "cpu")

def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1, use_snorm=False):
    if use_snorm:
        return nn.utils.spectral_norm(nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2) + dilation - 1, bias=bias, dilation=dilation))
    else:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2) + dilation - 1, bias=bias, dilation=dilation)


def default_conv1(in_channels, out_channels, kernel_size, bias=True, groups=3, use_snorm=False):
    if use_snorm:
        return nn.utils.spectral_norm(nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2), bias=bias, groups=groups))
    else:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2), bias=bias, groups=groups)


def default_conv3d(in_channels, out_channels, kernel_size, t_kernel=3, bias=True, dilation=1, groups=1,
                   use_snorm=False):
    if use_snorm:
        return nn.utils.spectral_norm(nn.Conv3d(
            in_channels, out_channels, (t_kernel, kernel_size, kernel_size), stride=1,
            padding=(0, kernel_size // 2, kernel_size // 2), bias=bias, dilation=dilation, groups=groups))
    else:
        return nn.Conv3d(
            in_channels, out_channels, (t_kernel, kernel_size, kernel_size), stride=1,
            padding=(0, kernel_size // 2, kernel_size // 2), bias=bias, dilation=dilation, groups=groups)


# def shuffle_channel()

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


def pixel_down_shuffle(x, downsacale_factor):
    batchsize, num_channels, height, width = x.size()

    out_height = height // downsacale_factor
    out_width = width // downsacale_factor
    input_view = x.contiguous().view(batchsize, num_channels, out_height, downsacale_factor, out_width,
                                     downsacale_factor)

    num_channels *= downsacale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()

    return unshuffle_out.view(batchsize, num_channels, out_height, out_width)


def sp_init(x):
    x01 = x[:, :, 0::2, :]
    x02 = x[:, :, 1::2, :]
    x_LL = x01[:, :, :, 0::2]
    x_HL = x02[:, :, :, 0::2]
    x_LH = x01[:, :, :, 1::2]
    x_HH = x02[:, :, :, 1::2]

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def dwt_init3d(x):
    x01 = x[:, :, :, 0::2, :] / 2
    x02 = x[:, :, :, 1::2, :] / 2
    x1 = x01[:, :, :, :, 0::2]
    x2 = x02[:, :, :, :, 0::2]
    x3 = x01[:, :, :, :, 1::2]
    x4 = x02[:, :, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def dwt_init_fine_grained(x, n):
    out = x
    for i in range(n):
        out = dwt_init(out)
    return out


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32, kernel_size=3, bias=True, use_snorm=False):
        super(ResidualDenseBlock, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        if use_snorm:
            self.conv1 = nn.utils.spectral_norm(nn.Conv2d(nf, gc, 3, 1, 1, bias=bias))
            self.conv2 = nn.utils.spectral_norm(nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias))
            self.conv3 = nn.utils.spectral_norm(nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias))
            self.conv4 = nn.utils.spectral_norm(nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias))
            self.conv5 = nn.utils.spectral_norm(nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias))
        else:
            self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
            self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
            self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
            self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
            self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        # x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32, use_snorm=False):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc, use_snorm)
        # self.RDB2 = ResidualDenseBlock(nf, gc)
        # self.RDB3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        # out = self.RDB2(out)
        # out = self.RDB3(out)
        return out * 0.2 + x


class RRDBblock(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32, nb=23, use_snorm=False):
        super(RRDBblock, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc, use_snorm=use_snorm)

        self.RRDB_trunk = mutil.make_layer(RRDB_block_f, nb)
        if use_snorm:
            self.trunk_conv = nn.utils.spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
        else:
            self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):

        return self.trunk_conv(self.RRDB_trunk(x))


class Channel_Shuffle(nn.Module):
    def __init__(self, conv_groups):
        super(Channel_Shuffle, self).__init__()
        self.conv_groups = conv_groups
        self.requires_grad = False

    def forward(self, x):
        return channel_shuffle(x, self.conv_groups)


class SP(nn.Module):
    def __init__(self):
        super(SP, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return sp_init(x)


class Pixel_Down_Shuffle(nn.Module):
    def __init__(self):
        super(Pixel_Down_Shuffle, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return pixel_down_shuffle(x, 2)


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class DWT_P(nn.Module):
    def __init__(self):
        super(DWT_P, self).__init__()
        self.requires_grad = False

    def forward(self, x, n):
        return dwt_init_fine_grained(x, n)


class DWT3d(nn.Module):
    def __init__(self):
        super(DWT3d, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init3d(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False
        if sign == -1:
            self.create_graph = False
            self.volatile = True


class MeanShift2(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift2, self).__init__(4, 4, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(4).view(4, 4, 1, 1)
        self.weight.data.div_(std.view(4, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False
        if sign == -1:
            self.volatile = True


class BasicBlock(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=False, act=nn.LeakyReLU(True), use_snorm=False):

        if use_snorm:
            m = [nn.utils.spectral_norm(nn.Conv2d(
                in_channels, out_channels, kernel_size,
                padding=(kernel_size // 2), stride=stride, bias=bias))
            ]
        else:
            m = [nn.Conv2d(
                in_channels, out_channels, kernel_size,
                padding=(kernel_size // 2), stride=stride, bias=bias)
            ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)


class Block3d(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size, t_kernel=3,
            bias=True, act=nn.LeakyReLU(True), res_scale=1, use_snorm=False):
        super(Block3d, self).__init__()
        m = []

        m.append(default_conv3d(in_channels, out_channels, kernel_size, bias=bias, use_snorm=use_snorm))
        m.append(act)
        m.append(default_conv3d(out_channels, out_channels, kernel_size, bias=bias, use_snorm=use_snorm))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x


class BBlock(nn.Module):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.LeakyReLU(True), res_scale=1, use_snorm=False):
        super(BBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, use_snorm=use_snorm))

        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x


class DBlock_com(nn.Module):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.LeakyReLU(True), res_scale=1, use_snorm=False):

        super(DBlock_com, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2, use_snorm=use_snorm))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=3, use_snorm=use_snorm))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x


class DBlock_inv(nn.Module):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.LeakyReLU(True), res_scale=1, use_snorm=False):

        super(DBlock_inv, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=3, use_snorm=use_snorm))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2, use_snorm=use_snorm))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x


class DBlock_com1(nn.Module):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.LeakyReLU(True), res_scale=1, use_snorm=False):

        super(DBlock_com1, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2, use_snorm=use_snorm))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(out_channels, out_channels, kernel_size, bias=bias, dilation=1, use_snorm=use_snorm))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x


class DBlock_inv1(nn.Module):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.LeakyReLU(True), res_scale=1, use_snorm=False):

        super(DBlock_inv1, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2, use_snorm=use_snorm))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(out_channels, out_channels, kernel_size, bias=bias, dilation=1, use_snorm=use_snorm))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x


class DBlock_com2(nn.Module):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.LeakyReLU(True), res_scale=1, use_snorm=False):

        super(DBlock_com2, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2, use_snorm=use_snorm))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2, use_snorm=use_snorm))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x


class DBlock_inv2(nn.Module):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.LeakyReLU(True), res_scale=1, use_snorm=False):

        super(DBlock_inv2, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2, use_snorm=use_snorm))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2, use_snorm=use_snorm))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x


class ShuffleBlock(nn.Module):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.LeakyReLU(True), res_scale=1, conv_groups=1, use_snorm=False):
        super(ShuffleBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, use_snorm=use_snorm))
        m.append(Channel_Shuffle(conv_groups))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x).mul(self.res_scale)
        return x


class DWBlock(nn.Module):
    def __init__(
            self, conv, conv1, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.LeakyReLU(True), res_scale=1, use_snorm=False):

        super(DWBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, use_snorm=use_snorm))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)

        m.append(conv1(in_channels, out_channels, 1, bias=bias, use_snorm=use_snorm))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x).mul(self.res_scale)
        return x


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.LeakyReLU(True), res_scale=1, use_snorm=False):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias, use_snorm=use_snorm))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Block(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.LeakyReLU(True), res_scale=1, use_snorm=False):

        super(Block, self).__init__()
        m = []
        for i in range(4):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias, use_snorm=use_snorm))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        # res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True, use_snorm=False):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias, use_snorm=use_snorm))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias, use_snorm=use_snorm))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class VGG_conv0(nn.Module):
    def __init__(self, in_nc, nf):
        super(VGG_conv0, self).__init__()
        # [64, 128, 128]
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [64, 64, 64]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [128, 32, 32]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        # [256, 16, 16]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [512, 8, 8]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)

        # self.avg_pool = nn.AvgPool2d(3, stride=2, padding=0, ceil_mode=True)  # /2
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))

        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))
        # fea = self.avg_pool(fea)

        return fea


class VGG_conv1(nn.Module):
    def __init__(self, in_nc, nf):
        super(VGG_conv1, self).__init__()
        # [64, 128, 128]
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [64, 64, 64]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [128, 32, 32]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        # [256, 16, 16]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [512, 8, 8]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)

        # self.avg_pool = nn.AvgPool2d(2, stride=1, padding=0, ceil_mode=True)  # /2
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))
        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))
        # fea = self.avg_pool(fea)

        return fea


class VGG_conv2(nn.Module):
    def __init__(self, in_nc, nf):
        super(VGG_conv2, self).__init__()
        # [64, 128, 128]
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [64, 64, 64]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [128, 32, 32]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        # [256, 16, 16]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [512, 8, 8]
        # self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        # self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        # self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        # self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)

        # self.avg_pool = nn.AvgPool2d(3, stride=2, padding=0, ceil_mode=True)  # /2
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))

        # fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        # fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))
        # fea = self.avg_pool(fea)

        return fea
