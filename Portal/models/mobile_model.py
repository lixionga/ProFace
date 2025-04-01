import torch
import torch.nn as nn
import torch.nn.functional as F
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FaceSwap(nn.Module):
    def __init__(self, use_gpu=True):
        super().__init__()
        self.swap_model = UNet()
        self.swap_model.eval()

    def set_model_param(self, id_emb, id_feature_map, model_weight=None):
        predict_model = BuildFaceSwap()
        if model_weight is not None:
            predict_model.load_state_dict(model_weight)

        predict_model.eval()
        # predict_model.to(device)
        weights_encoder, weights_decoder, encode_mod, decode_mod = predict_model(id_emb, id_feature_map)
        for i in range(len(self.swap_model.Encoder)):
            self.swap_model.Encoder[i][0].weight.data.copy_(weights_encoder[i][0].unsqueeze(1))
            self.swap_model.Encoder[i][1].weight.data.copy_(encode_mod[i])

        for i in range(len(self.swap_model.Decoder)):
            self.swap_model.Decoder[i][0].weight.data.copy_(weights_decoder[i][0].unsqueeze(1))
            self.swap_model.Decoder[i][1].weight.data.copy_(decode_mod[i])
        self.swap_model.mask = predict_model.mask
        self.swap_model.final = predict_model.final

    def forward(self, att_img):
        img, mask = self.swap_model(att_img)
        return img, mask

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Encoder_channel = [3, 32, 64, 128, 256, 512]
        self.Encoder = nn.ModuleList()#用于管理子层的列表
        for i in range(len(self.Encoder_channel)-1):         
            self.Encoder.append(nn.Sequential(*[
                nn.Conv2d(self.Encoder_channel[i], self.Encoder_channel[i], kernel_size=4, stride=2, padding=1, groups=self.Encoder_channel[i],bias=False),
                nn.Conv2d(self.Encoder_channel[i], self.Encoder_channel[i+1], kernel_size=1,bias=False),

            ]))

        self.Decoder_inchannel = [512, 512, 256, 128]
        self.Decoder_outchannel = [256, 128, 64, 32]
        self.Decoder = nn.ModuleList()
        for i in range(len(self.Decoder_inchannel)):            
            self.Decoder.append(nn.Sequential(*[
                nn.Conv2d(self.Decoder_inchannel[i], self.Decoder_inchannel[i], kernel_size=3, stride=1, padding=1, groups=self.Decoder_inchannel[i],bias=False),
                nn.Conv2d(self.Decoder_inchannel[i], self.Decoder_outchannel[i], kernel_size=1,bias=False),
            ]))

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

        self.relu = nn.LeakyReLU(0.1)
        self.up = nn.Upsample(scale_factor=2.0, align_corners=True, mode='bilinear')

    def forward(self, data):
        
        x = (data - 0.5) / 0.5
        arr_x = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            x = self.relu(x)
            arr_x.append(x)
        mask = x.detach()
        for i in range(len(self.mask)):
            mask = self.mask[i](mask)
        y = arr_x[-1]
        for i in range(len(self.Decoder)):
            y = self.up(y)
            y = self.relu(self.Decoder[i](y))
            if i != len(self.Decoder) - 1:
                print(y.shape)
                print(arr_x[len(self.Decoder)-1-i].shape)
                y = torch.cat((y, arr_x[len(self.Decoder)-1-i]), 1)
        # print(y)
        out = self.final(y)
        out = (1 + out) / 2.0
        out = out * mask + (1 - mask) * data
        return out, mask



class BuildFaceSwap(nn.Module):
    def __init__(self,):
        super(BuildFaceSwap, self).__init__()
        encoder_scale = 2

        self.Encoder_channel = [3, 64//encoder_scale, 128//encoder_scale, 256//encoder_scale, 512//encoder_scale, 1024//encoder_scale]

        self.EncoderModulation = nn.ModuleList()
        for i in range(len(self.Encoder_channel)-1):
            self.EncoderModulation.append(Mod2Weight(self.Encoder_channel[i], self.Encoder_channel[i+1]))

        self.Decoder_inchannel = [1024//encoder_scale, 1024//encoder_scale, 512//encoder_scale, 256//encoder_scale]
        self.Decoder_outchannel = [512//encoder_scale, 256//encoder_scale, 128//encoder_scale, 64//encoder_scale]
        
        self.DecoderModulation = nn.ModuleList()
        for i in range(len(self.Decoder_inchannel)):
            self.DecoderModulation.append(Mod2Weight(self.Decoder_inchannel[i], self.Decoder_outchannel[i]))

        self.predictor = WeightPrediction(self.Encoder_channel[:-1], self.Decoder_inchannel)

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
        self.relu = nn.LeakyReLU(0.1)
        self.up = nn.Upsample(scale_factor=2.0, align_corners=True, mode='bilinear')


        mask_channel = [1024//encoder_scale, 256//encoder_scale, 128//encoder_scale, 64//encoder_scale, 16//encoder_scale, 4//encoder_scale]
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


    def forward(self, id_emb, id_feature_map):
        weights_encoder, weights_decoder = self.predictor(id_feature_map)

        encode_mod = []
        decode_mod = []
        for i in range(len(self.EncoderModulation)):
            encode_mod.append(self.EncoderModulation[i](id_emb))

        for i in range(len(self.DecoderModulation)):
            decode_mod.append(self.DecoderModulation[i](id_emb))

        return weights_encoder, weights_decoder, encode_mod, decode_mod


class WeightPrediction(nn.Module):#预测解码器编码器权重
    def __init__(self, encoder_channels, decoder_channels, style_dim=512):
        super().__init__()
        self.first = nn.Conv2d(style_dim, style_dim, kernel_size=4, stride=1)
        self.first_decoder = nn.Conv2d(style_dim, style_dim, kernel_size=2, stride=1)

        self.decoder_norm = nn.BatchNorm2d(style_dim)
        self.norm =nn.BatchNorm2d(style_dim)
        
        self.relu = nn.LeakyReLU(0.1)

        encoder_channels += [style_dim]
        encoder_channels = encoder_channels[::-1]
        self.encoder = nn.ModuleList()
        for i in range(len(encoder_channels)-1):
            self.encoder.append(ConvBlock(encoder_channels[i], encoder_channels[i+1]))
        
        decoder_channels = [style_dim] + decoder_channels
        self.decoder = nn.ModuleList()
        for i in range(len(decoder_channels)-1):
            self.decoder.append(ConvBlock(decoder_channels[i], decoder_channels[i+1]))

    def forward(self, z_id):
        encoder_weights = []
        decoder_weights = []
        z_id = self.first(z_id)
        z_id = self.relu(self.norm(z_id))
        x = z_id
        for i in range(len(self.encoder)):                
            x, weight = self.encoder[i](x)
            encoder_weights.append(weight)
        y = z_id
        y = self.relu(self.decoder_norm(self.first_decoder(y)))
        for i in range(len(self.decoder)):
            y, weight = self.decoder[i](y)
            decoder_weights.append(weight)
        return encoder_weights[::-1], decoder_weights


class Mod2Weight(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim=512):
        super().__init__()
        self.out_channel = out_channel

        self.kernel = 1
        self.stride = 1
        self.eps = 1e-16

        self.style = nn.Linear(style_dim, in_channel,dtype=torch.float32)
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, self.kernel, self.kernel).float())#方便地创建可训练的参数变量，并将其添加到模型中。通常，神经网络中的权重和偏置是需要被训练的参数。


    def forward(self, style, b=1):
        style = self.style(style)
        scale_deta = style.unsqueeze(dim=1).unsqueeze(dim=-1).unsqueeze(dim=-1)

        weights = self.weight.unsqueeze(dim=0) * (scale_deta + 1) # * 是前后逐元素相乘

        d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)  #rsqrt（x）计算1/x½
        weights = weights * d

        _, _, *ws = weights.shape
        
        weights = weights.reshape((b * self.out_channel, *ws))
        return weights




class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, padding_mode='zeros'):
        super().__init__()
        self.out_channel = out_channel
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU(0.1)
        self.weight = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x): 
        out = self.relu(self.norm(self.conv(x)))
        weight = self.weight(out)
        return out, weight


class Conv2dFunction(nn.Module):
    def __init__(self,stride=1,padding=0):
        super().__init__()


        self.stride = stride
        self.padding = padding


    def set_weight(self,cond_weights):
        self.cond_weights = cond_weights


    def forward(self,x):
        # bs,_,h,w = inputs.shape

        out = F.conv2d(x,weight=self.cond_weights,bias=None,stride=self.stride,padding=self.padding)
        

        return out

class FunctionConv2d(nn.Module):
    def __init__(self,stride=2,padding=1,groups=1):
        super().__init__()


        self.stride = stride
        self.padding = padding
        self.groups = groups

    def set_weight(self,cond_weights):
        self.cond_weights = cond_weights

    def forward(self,x):
        
        out = F.conv2d(x,weight=self.cond_weights,bias=None,stride=self.stride,padding=self.padding,groups=self.groups)

        return out


def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.divide(input, norm)
    return output
