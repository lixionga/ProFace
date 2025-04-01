import torch
import torch.nn as nn
import torch.nn.functional as F



class FaceSwap(nn.Module):
    def __init__(self, use_gpu=True):
        super().__init__()
        self.swap_model = UNet()

    def forward(self, att_img,id_emb):
        out, mask = self.swap_model(att_img,id_emb)

        return out, mask


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Encoder_channel = [3, 32, 64, 128, 256, 512]
        self.Encoder = nn.ModuleList()#用于管理子层的列表
        for i in range(len(self.Encoder_channel)-1):
            self.Encoder.append(nn.Sequential(*[
                nn.Conv2d(self.Encoder_channel[i], self.Encoder_channel[i], kernel_size=4, stride=2, padding=1, groups=self.Encoder_channel[i]),
                nn.Conv2d(self.Encoder_channel[i], self.Encoder_channel[i+1], kernel_size=1),
            ]))

        self.Decoder_inchannel = [512, 512, 256, 128]
        self.Decoder_outchannel = [256, 128, 64, 32]
        self.Decoder = nn.ModuleList()
        for i in range(len(self.Decoder_inchannel)):            
            self.Decoder.append(nn.Sequential(*[
                nn.Conv2d(self.Decoder_inchannel[i], self.Decoder_inchannel[i], kernel_size=3, stride=1, padding=1, groups=self.Decoder_inchannel[i]),
                nn.Conv2d(self.Decoder_inchannel[i], self.Decoder_outchannel[i], kernel_size=1),
            ]))

        self.relu = nn.LeakyReLU(0.1)
        self.up = nn.Upsample(scale_factor=2.0, align_corners=True, mode='bilinear')


        self.final = nn.Sequential(*[
                nn.Upsample(scale_factor=2.0, align_corners=True, mode='bilinear'), 
                nn.Conv2d(self.Decoder_outchannel[-1], self.Decoder_outchannel[-1] // 4, kernel_size=1),
                nn.BatchNorm2d(self.Decoder_outchannel[-1] // 4), 
                nn.LeakyReLU(0.1),
                nn.Conv2d(self.Decoder_outchannel[-1] // 4, 3, 3, padding=1),
                nn.BatchNorm2d(3), 
                nn.LeakyReLU(0.1),
                nn.Conv2d(3, 3, 3, padding=1),
                nn.Tanh()
        ])


        mask_channel = [512, 128, 64, 32, 8, 2]
        mask = []
        for i in range(len(mask_channel)-1):
            mask.append(nn.Sequential(
                    nn.Upsample(scale_factor=2.0, align_corners=True, mode='bilinear'), 
                    nn.Conv2d(mask_channel[i], mask_channel[i], kernel_size=3, stride=1, padding=1, groups=mask_channel[i]),
                    nn.Conv2d(mask_channel[i], mask_channel[i+1], kernel_size=1, stride=1),
                    nn.BatchNorm2d(mask_channel[i+1]), 
                    nn.LeakyReLU(0.1)
                ))
        mask.append(nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1))
        mask.append(nn.Sigmoid())
        self.mask = nn.Sequential(*mask)

        # style_inchannel = [512,256,128,64]
        # style_outchannel = [512,256,128,64]

        style = []
        for i in range(3):
            style.append(AdaINBlock(512,512))
        self.style = nn.Sequential(*style)


    def forward(self, data,id_emb):
        x = (data - 0.5) / 0.5
        arr_x = []
        for i in range(len(self.Encoder)):
            x = self.relu(self.Encoder[i](x))
            arr_x.append(x)

        mask = x.detach()

        for i in range(len(self.mask)):
            mask = self.mask[i](mask)

        # y = self.style[0](arr_x[-1],id_emb)

        y = arr_x[-1]
        for i in range(len(self.style)):
            y = self.style[i](y,id_emb)

        for i in range(len(self.Decoder)):
            y = self.up(y)
            y = self.relu(self.Decoder[i](y))
            if i != len(self.Decoder) - 1:
                # weight = self.style[i+1](arr_x[len(self.Decoder)-1-i] ,id_emb)
                y = torch.cat((y, arr_x[len(self.Decoder)-1-i]), 1)
        out = self.final(y)
        out = (1 + out) / 2.0
        out = out * mask + (1-mask) * data
        return out, mask


class ApplyStyle(nn.Module):
    def __init__(self, latent_size, channels):
        super().__init__()
        self.linear = nn.Linear(latent_size, channels * 2)

    def forward(self, x, latent):
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)    # [batch_size, 2, n_channels, ...]
        #x = x * (style[:, 0] + 1.) + style[:, 1]
        x = x * (style[:, 0] * 1 + 1.) + style[:, 1] * 1

        return x
class AdaINBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        # self.dw_conv = nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1,groups=in_channel)
        self.style1 = ApplyStyle(512,out_channel)
        self.style2 = ApplyStyle(512,out_channel)
        self.conv1 = nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1)
        self.act = nn.ReLU(True)

    def forward(self,x,id_emb):

        # y = self.style1(x,id_emb)
        # y = self.dw_conv(y)
        # y = self.style2(y,id_emb)
        # out = x +y

        y = self.conv1(x)
        y = self.style1(y,id_emb)
        y = self.act(y)
        y = self.conv2(y)
        y = self.style2(y,id_emb)
        out = y + x

        return out


class Conv2dFunction(nn.Module):
    def __init__(self,stride,padding,groups=1):
        super().__init__()

        self.stride = stride
        self.padding = padding
        self.groups = groups

    def set_weight(self,weight):
        self.weight = weight

    def forward(self,x):

        out = F.conv2d(x,weight=self.weight,bias=None,stride=self.stride,padding=self.padding,groups=self.groups)

        return out

class Spatial_attention(nn.Module):
    def __init__(self,kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(2,1,kernel_size,padding=1,bias=False)
    def forward(self,x):
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out,_ = torch.max(x,dim=1,keepdim=True)

        x = torch.cat([avg_out,max_out],dim=1)
        x = self.conv1(x)
        return x


def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.divide(input, norm)
    return output
