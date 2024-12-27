import sys
sys.path.append('/data/lk/ID-disen4/models/deghosting')
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from stylegan2 import StyleGANv2Generator
from mmcv.cnn import xavier_init

class ResidualDenseBlock(nn.Module):
    def __init__(self, mid_channels=64, growth_channels=32):
        super().__init__()
        for i in range(5):
            out_channels = mid_channels if i == 4 else growth_channels
            self.add_module(
                f'conv{i + 1}',
                nn.Conv2d(mid_channels + i * growth_channels, out_channels, 3,
                          1, 1))
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.scale = 0.2

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.relu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.relu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * self.scale + x

class RRDB(nn.Module):
    def __init__(self, mid_channels, growth_channels=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(mid_channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(mid_channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(mid_channels, growth_channels)
        self.scale = 0.2
    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * self.scale + x

class RRDBNet(nn.Module):
    def __init__(self,
                 num_in=3,
                 num_mid=64,
                 num_out=256,
                 num_blocks=23,
                 num_grow=32):

        super().__init__()

        self.conv_first = nn.Conv2d(num_in, num_mid, 3, 1, 1)
        self.body = self._make_layer(
            num_blocks,
            mid_channels=num_mid,
            growth_channels=num_grow)
        self.conv_body = nn.Conv2d(num_mid, num_mid, 3, 1, 1)

        self.conv_up = nn.Conv2d(num_mid, num_out, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def _make_layer(self, num_blocks, mid_channels, growth_channels):
        layers = []
        for _ in range(num_blocks):
            layers.append(RRDB(mid_channels, growth_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.conv_up(feat)
        out = self.lrelu(feat)
        return out

class PixelShufflePack(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)

    def init_weights(self):
        xavier_init(self.upsample_conv, distribution='uniform')

    def forward(self, x):
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x

class FCN(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 img_channels=3,
                 rrdb_channels=64,
                 num_rrdbs=23,
                 style_channels=512,
                 ):
        super(FCN, self).__init__()

        self.in_size = in_size
        self.style_channels = style_channels
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 512,
            128: 256,
            256: 128,
            512: 64,
            1024: 32,
        }
        # encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(
            RRDBNet(img_channels, rrdb_channels, num_blocks=num_rrdbs, num_out=channels[in_size]),
        )
        for c in [2 ** i for i in range(int(math.log(in_size, 2)), 1, -1)]:
            in_channels = channels[c]
            if c > 4:
                out_channels = channels[c // 2]
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(0.2, True))
            else:
                block = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(0.2, True),
                    nn.Flatten(),
                    nn.Linear(16 * in_channels, (int(math.log(out_size, 2)) * 2 - 2) * style_channels))
            self.encoder.append(block)

        # additional modules for StyleGANv2
        self.fusion_out = nn.ModuleList()
        self.fusion_skip = nn.ModuleList()
        for c in [2 ** i for i in range(2, int(math.log(in_size, 2)) + 1)]:
            num_channels = channels[c]
            self.fusion_out.append(
                nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1, bias=True))
            self.fusion_skip.append(
                nn.Conv2d(num_channels + 3, 3, 3, 1, 1, bias=True))


        self.decoder = nn.ModuleList()
        for c in [2**i for i in range(int(math.log(in_size, 2)), int(math.log(out_size, 2) + 1))]:
            if c == in_size:
                in_channels = channels[c]
            else:
                in_channels = 2 * channels[c]

            if c < out_size:
                out_channels = channels[c * 2]
                self.decoder.append(
                    PixelShufflePack(
                        in_channels, out_channels, 2, 3))
            else:
                self.decoder.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, 64, 3, 1, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(64, img_channels, 3, 1, 1)))


class Deghosting(nn.Module):
    def __init__(self, in_size, out_size, pretrain=None):
        super(Deghosting, self).__init__()
        self.FCN = FCN(in_size=in_size, out_size=out_size, style_channels=512)
        self.generator = StyleGANv2Generator(out_size=out_size, style_channels=512, pretrained=pretrain)
        for p in self.generator.parameters():
            p.requires_grad = False

    def forward(self, lq):
        h, w = lq.shape[2:]
        if h != self.FCN.in_size or w != self.FCN.in_size:
            raise AssertionError(
                f'Spatial resolution must equal in_size ({self.FCN.in_size}).'
                f' Got ({h}, {w}).')

        # encoder
        feat = lq
        encoder_feats = []
        for block in self.FCN.encoder:
            feat = block(feat)
            encoder_feats.append(feat)
        encoder_feats = encoder_feats[::-1]

        latent = encoder_feats[0].view(lq.size(0), -1, self.FCN.style_channels)
        encoder_feats = encoder_feats[1:]


        # generator
        injected_noise = [
            getattr(self.generator, f'injected_noise_{i}')
            for i in range(self.generator.num_injected_noises)
        ]

        out = self.generator.constant_input(latent)
        out = self.generator.conv1(out, latent[:, 0], noise=injected_noise[0])
        skip = self.generator.to_rgb1(out, latent[:, 1])

        _index = 1


        generator_feats = []
        for up_conv, conv, noise1, noise2, to_rgb in zip(
                self.generator.convs[::2], self.generator.convs[1::2],
                injected_noise[1::2], injected_noise[2::2],
                self.generator.to_rgbs):


            if out.size(2) <= self.FCN.in_size:
                fusion_index = (_index - 1) // 2
                feat = encoder_feats[fusion_index]

                out = torch.cat([out, feat], dim=1)
                out = self.FCN.fusion_out[fusion_index](out)

                skip = torch.cat([skip, feat], dim=1)
                skip = self.FCN.fusion_skip[fusion_index](skip)


            out = up_conv(out, latent[:, _index], noise=noise1)
            out = conv(out, latent[:, _index + 1], noise=noise2)
            skip = to_rgb(out, latent[:, _index + 2], skip)


            if out.size(2) > self.FCN.in_size:
                generator_feats.append(out)

            _index += 2

        # decoder
        hr = encoder_feats[-1]
        for i, block in enumerate(self.FCN.decoder):
            if i > 0:
                hr = torch.cat([hr, generator_feats[i - 1]], dim=1)
            hr = block(hr)

        return hr



if __name__ == '__main__':
    import cv2 
    import torchvision
    from torchvision import transforms
    totensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([128,128])
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
    input_path = '/home/yl/lk/code/ID-dise4/out/test_INN8/anonymous_fir_imgs/'
    out_path = '/home/yl/lk/code/ID-dise4/out/test_INN8/anonymous_fir_degh_imgs/'

    degh_net = Deghosting(in_size=128, out_size=256, pretrain='/home/yl/lk/code/ID-dise-psp/550000.pt')
    degh_net.load_state_dict(torch.load('/userHOME/yl/lk_data/deghosting_weight/deghosting_weight_1_140.pth'))
    degh_net = degh_net.to('cuda:0')
    degh_net = degh_net.eval()
    with torch.no_grad():
        for test_idx in range(4):
            raw_img = cv2.imread(input_path+str(test_idx)+'.png')
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            raw_img = cv2.resize(raw_img,(256,256),interpolation=cv2.INTER_CUBIC)
            raw_img = totensor(raw_img)
            bg_img = raw_img.unsqueeze(0).to('cuda:0')
            out_img = degh_net(bg_img)    
            torchvision.utils.save_image(out_img,out_path+str(test_idx)+'.png')
            
