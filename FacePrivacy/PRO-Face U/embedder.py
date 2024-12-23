import torch.optim
import torch.nn as nn
from hinet import Hinet
import modules.Unet_common as common
import config.config as c


# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device = c.DEVICE

dwt = common.DWT()
iwt = common.IWT()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = Hinet()

    def forward(self, input1, input2, password ,rev=False):
        if not rev:
            secret_img, cover_img = input1, input2
            # cover_input = dwt(cover_img)  # torch.Size([8, 12, W, H])
            # secret_input = dwt(secret_img)
            # input_img = torch.cat((cover_input, secret_input), 1)
            input_img = torch.cat((cover_img, secret_img), 1)
            output = self.model(input_img,password)  # torch.Size([8, 24, W, H])
            # output_steg = output.narrow(1, 0, 4 * c.channels_in)  # 取前半部分通道
            # output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)  # 取后半部分通道
            output_steg = output.narrow(1, 0, c.channels_in)  # 取前半部分通道
            output_z = output.narrow(1, c.channels_in, output.shape[1] - c.channels_in)  # 取后半部分通道
            # output_steg_img = iwt(output_steg)
            # return output_z, output_steg, output_steg_img
            # return output_z, output_steg, output_steg_img
            return output_z, output_steg
        else:
            output_z, output_steg = input1, input2
            output_rev = torch.cat((output_steg, output_z), 1)
            output_image = self.model(output_rev, rev=True)
            # secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
            # secret_rev_img = iwt(secret_rev)
            cover_rev_img = output_image.narrow(1, 0, c.channels_in)
            secret_rev_img = output_image.narrow(1, c.channels_in, output_image.shape[1] - c.channels_in)
            return secret_rev_img, cover_rev_img


class ModelDWT(nn.Module):
    def __init__(self, n_blocks=6):
        super(ModelDWT, self).__init__()
        self.model = Hinet()
        self.device = torch.device('cpu')

    def to(self, device):
        super(ModelDWT, self).to(device)
        self.device = device
        return self

    def forward(self, input1, input2, password, rev=False):
        if not rev:
            secret_img, cover_img = input1, input2
            cover_dwt = dwt(cover_img)  # torch.Size([Batch, 12, W, H])
            cover_dwtt = cover_dwt.narrow(1, 0, c.channels_in)
            secret_dwt = dwt(secret_img)
            secret_dwts = secret_dwt.narrow(1, 0, c.channels_in)
            input_dwt = torch.cat((cover_dwt, secret_dwt), 1)
            output = self.model(input_dwt,password)  # torch.Size([Batch, 24, W, H])
            output_steg_dwt = output.narrow(1, 0, 4 * c.channels_in)  # 取前半部分通道
            output_steg_low = output_steg_dwt.narrow(1, 0, c.channels_in )
            output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)  # 取后半部分通道
            # output_steg = output.narrow(1, 0, c.channels_in)  # 取前半部分通道
            # output_z = output.narrow(1, c.channels_in, output.shape[1] - c.channels_in)  # 取后半部分通道
            output_z_img = iwt(output_z, self.device)
            output_steg_img = iwt(output_steg_dwt, self.device)
            # return output_z, output_steg, output_steg_img
            # return output_z, output_steg, output_steg_img
            return output_steg_low,secret_dwts,output_z_img, output_steg_img
        else:
            output_z, output_steg_img = input1, input2
            output_steg_dwt = dwt(output_steg_img)
            output_rev = torch.cat((output_steg_dwt, output_z), 1)
            output_dwt = self.model(output_rev, password, rev=True)
            secret_rev_dwt = output_dwt.narrow(1, 4 * c.channels_in, output_dwt.shape[1] - 4 * c.channels_in)
            secret_rev_img = iwt(secret_rev_dwt, self.device)
            cover_rev_dwt = output_dwt.narrow(1, 0, 4 * c.channels_in)
            cover_rev_img = iwt(cover_rev_dwt, self.device)
            return secret_rev_img, cover_rev_img


class FaceClassifierHead(nn.Module):
    def __init__(self):
        super(FaceClassifierHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.head(x)


class UtilityConditioner(nn.Module):
    def __init__(self):
        super(UtilityConditioner, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
            nn.Linear(16, 64),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784)
        )

    def forward(self, x):
        return self.head(x)
class Noisemaker(nn.Module):
    def __init__(self):
        super(Noisemaker, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, x):
        return self.head(x)


# class Noisemaker(nn.Module):
#     def __init__(self, num_channels=3, num_filters=64):
#         super(Noisemaker, self).__init__()
#
#         self.encoder = nn.Sequential(
#             nn.Conv2d(num_channels, num_filters, kernel_size=7, stride=1, padding=3),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(inplace=True)
#         )
#
#         self.residual_blocks = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(num_filters * 4, num_filters * 4, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(num_filters * 4, num_filters * 4, kernel_size=3, padding=1)
#             ) for _ in range(3)
#         ])
#
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(num_filters * 4, num_filters * 2, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(num_filters * 2, num_filters, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_filters, num_channels, kernel_size=7, stride=1, padding=3),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         # Encoder
#         encoded = self.encoder(x)
#
#         # Residual blocks
#         for block in self.residual_blocks:
#             encoded = encoded + block(encoded)
#
#         # Decoder
#         output = self.decoder(encoded)
#
#         # Clip output to range [-1, 1]
#         output = torch.clamp(output, -1.0, 1.0)
#
#         # Scale output to range [0, 1]
#         output = (output + 1.0) / 2.0
#
#         return output
def init_model(mod, device):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            # param.data = c.init_scale * torch.randn(param.data.shape).cuda()
            param.data = c.init_scale * torch.randn(param.data.shape).to(device)
            if split[-2] == 'conv5':
                param.data.fill_(0.)