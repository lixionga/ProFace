import torch
import torch.nn as nn
import modules.module_util as mutil


# Dense connection
class ResidualDenseBlock_out(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualDenseBlock_out, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # initialization
        mutil.initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5

# Dense connection
# class ResidualDenseBlock_out(nn.Module):
#     def __init__(self, input, output, nf=3, gc=32, bias=True, use_snorm=False):
#         super(ResidualDenseBlock_out, self).__init__()
#         # gc: growth channel, i.e. intermediate channels
#         if use_snorm:
#             self.conv1 = nn.utils.spectral_norm(nn.Conv2d(nf, gc, 3, 1, 1, bias=bias))
#             self.conv2 = nn.utils.spectral_norm(nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias))
#             self.conv3 = nn.utils.spectral_norm(nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias))
#             self.conv4 = nn.utils.spectral_norm(nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias))
#             self.conv5 = nn.utils.spectral_norm(nn.Conv2d(nf + 4 * gc, output, 3, 1, 1, bias=bias))
#         else:
#             self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
#             self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
#             self.conv3 = nn.Conv2d(input + 2 * 32, 32, 3, 1, 1, bias=bias)
#             self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
#             self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
#         self.lrelu = nn.LeakyReLU(inplace=True)
#         # initialization
#         # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
#         mutil.initialize_weights([self.conv5], 0.)
#
#     def forward(self, x):
#         x1 = self.lrelu(self.conv1(x))
#         x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
#         x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
#         x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
#         x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
#         return x5
