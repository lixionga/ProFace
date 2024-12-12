import torch
import torch.nn as nn
from torchvision import transforms
from collections import OrderedDict


class DenseLayer2D(nn.Sequential):
    def __init__(self, c_in, grow_rate, bn_size, drop_rate):
        super(DenseLayer2D, self).__init__()
        grow_size = bn_size * grow_rate
        self.add_module('norm1', nn.BatchNorm2d(c_in, eps=1e-05, momentum=0.1, affine=True))
        self.add_module('relu1', nn.LeakyReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(c_in, grow_size, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.add_module('norm2', nn.BatchNorm2d(grow_size, eps=1e-05, momentum=0.1, affine=True))
        self.add_module('relu2', nn.LeakyReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(grow_size, grow_rate, kernel_size=(3, 3), stride=(1, 1),
                                           padding=(1, 1), bias=False))
        self.dropout = nn.Dropout(p=drop_rate)
        self.c_out = c_in + grow_rate

    def forward(self, x):
        new_features = super(DenseLayer2D, self).forward(x)
        new_features = self.dropout(new_features)
        return torch.cat([x, new_features], 1)


class DenseBlock2D(nn.Sequential):
    def __init__(self, nlayers, c_in, bn_size, grow_rate, drop_rate):
        super(DenseBlock2D, self).__init__()
        self.c_out = 0
        for i in range(nlayers):
            layer = DenseLayer2D(c_in + grow_rate * i, grow_rate, bn_size, drop_rate)
            self.add_module(f'denselayer{i+1}', layer)
            if i == nlayers - 1:
                self.c_out = layer.c_out


class DownsampleTransition2D(nn.Sequential):
    def __init__(self, c_in, c_out):
        super(DownsampleTransition2D, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(c_in, eps=1e-05, momentum=0.1, affine=True))
        self.add_module('relu', nn.LeakyReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.add_module('pool', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))


class UpsampleTransition2D(nn.Sequential):
    def __init__(self, c_in, c_out):
        super(UpsampleTransition2D, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(c_in, eps=1e-05, momentum=0.1, affine=True))
        self.add_module('relu', nn.LeakyReLU(inplace=True))
        self.add_module('deconv', nn.ConvTranspose2d(c_in, c_out, kernel_size=(2, 2), stride=(2, 2)))


class SiameseDenseUNets(nn.Module):
    def __init__(self, grow_rate=4, block_config=(3, 3, 3), ninit_feat=16, bn_size=4, drop_rate=0.25, c_in=3, c_out=3):
        super(SiameseDenseUNets, self).__init__()
        self.nblock = len(block_config)

        # First convolution
        self.initial = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(c_in, ninit_feat, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(ninit_feat)),
            ('relu0', nn.LeakyReLU(inplace=False))
        ]))

        # Encoding Denseblocks
        nfeat = ninit_feat
        ntrans_feats = []
        for i, nlayers in enumerate(block_config):
            dense_block = DenseBlock2D(nlayers, nfeat, bn_size, grow_rate, drop_rate)
            nfeat = dense_block.c_out
            trans_block = DownsampleTransition2D(nfeat, nfeat)
            setattr(self, f'encode_block{i}', dense_block)
            setattr(self, f'encode_trans{i}', trans_block)
            ntrans_feats.append(nfeat)

        # Bottom block
        self.bottom_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(nfeat, nfeat, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(nfeat)),
            ('relu0', nn.LeakyReLU(inplace=True))
        ]))

        # Decoding blocks
        for i, nlayers in enumerate(reversed(block_config)):
            setattr(self, f'decode_trans{i}', UpsampleTransition2D(nfeat, nfeat // 2))
            nfeat //= 2
            nfeat += ntrans_feats[-i-1]
            dense_block = DenseBlock2D(nlayers, nfeat, bn_size, grow_rate, drop_rate)
            nfeat = dense_block.c_out
            setattr(self, f'decode_block{i}', dense_block)

            # Final pool
            unpool = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(nfeat, c_out, kernel_size=(3, 3),
                                    stride=(1, 1), padding=(1, 1), bias=False)),
                ('relu0', nn.LeakyReLU(inplace=True)),
                ('pool0', nn.ConvTranspose2d(
                    c_out, c_out, kernel_size=2 ** (block_config[-1]-i-1),
                    stride=2 ** (block_config[-1]-i-1), padding=(0, 0)))
            ]))
            setattr(self, f'unpool{i}', unpool)

        # Combine input with output to preserve spacial resolution
        self.final_conv_1x1 = nn.Sequential(OrderedDict([
            ('conv0', nn.ConvTranspose2d(6, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        ]))
        self.thresh = nn.Sequential(nn.Threshold(threshold=0.5, value=0.0))

    def init_x(self, x):
        return self.initial(x)

    def encoder(self, x, i):
        down_dense = getattr(self, f"encode_block{i}")
        down_trans = getattr(self, f"encode_trans{i}")
        encode_outs = down_dense(x)
        x0 = down_trans(encode_outs)
        return x0, encode_outs

    def bottom(self, x):
        return self.bottom_block(x)

    def decoder(self, x, encode_outs, i):
        up_trans = getattr(self, f"decode_trans{i}")
        up_dense = getattr(self, f"decode_block{i}")
        unpool = getattr(self, f"unpool{i}")
        x0 = up_trans(x)
        # Concate the encoder feature channels
        x1 = torch.cat([x0, encode_outs[-i-1]], 1)
        x2 = up_dense(x1)
        out = unpool(x2)
        return x2, out

    def final_conv(self, x):
        return self.final_conv_1x1(x)

    def threshold(self, x):
        return self.thresh(x)


class ProFaceEmbedder(nn.Module):
    def __init__(self, c_in=3, block_config=(3, 3, 3)):
        super(ProFaceEmbedder, self).__init__()
        self.nblocks = len(block_config)
        self.oImgNet = SiameseDenseUNets(
            grow_rate=4, block_config=block_config, ninit_feat=16, bn_size=4,
            drop_rate=0.25, c_in=c_in, c_out=c_in
        )
        self.tImgNet = SiameseDenseUNets(
            grow_rate=4, block_config=block_config, ninit_feat=16, bn_size=4,
            drop_rate=0.25, c_in=c_in, c_out=c_in
        )

    def forward(self, orig_img, targ_img):
        # initial block
        o = self.oImgNet.init_x(orig_img)
        t = self.tImgNet.init_x(targ_img)
        encode_outs_o = []
        encode_outs_t = []
        # x = init_o + init_t

        # Encoding blocks
        for i in range(self.nblocks):
            o, o_enc_out = self.oImgNet.encoder(o, i)
            t, t_enc_out = self.tImgNet.encoder(t, i)
            # x = o_enc + t_enc
            encode_outs_o.append(o_enc_out)
            encode_outs_t.append(t_enc_out)

        # Bottom block
        x_o = self.oImgNet.bottom(o)
        x_t = self.tImgNet.bottom(t)
        x = x_o + x_t

        # Decoding block
        o_outs, t_outs = None, None
        for i in range(self.nblocks):
            o_dec, o_dec_out = self.oImgNet.decoder(x, encode_outs_o, i)
            t_dec, t_dec_out = self.tImgNet.decoder(x, encode_outs_t, i)
            x = o_dec + t_dec
            # Concate feature maps
            if i == 0:
                o_outs = o_dec_out.clone()
                t_outs = t_dec_out.clone()
            else:
                o_outs += o_dec_out
                t_outs += t_dec_out

        # Combine original and target image
        image_out = o_outs + t_outs
        return image_out


if __name__ == "__main__":
    from torchsummary import summary
    x = torch.rand([8, 3, 224, 224])
    y = torch.rand([8, 3, 224, 224])
    embedder = ProFaceEmbedder(c_in=3, block_config=(3, 3, 3))
    z = embedder(x, y)
    print(z.shape)
    assert z.shape == x.shape
    summary(embedder, [(3, 224, 224), (3, 224, 224)], device='cpu')
