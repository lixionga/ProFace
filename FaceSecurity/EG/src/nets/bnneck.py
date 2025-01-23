import torch.nn as nn
import torch
import torch.nn.functional as F

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class BNClassifier(nn.Module):
    '''bn + fc'''

    def __init__(self, in_dim, class_num):
        super(BNClassifier, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)

        self.bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    # def forward(self, x):
    #     feature = self.bn(x)
    #     if not self.training:
    #         return feature, None
    #     cls_score = self.classifier(feature)
    #     return feature, cls_score
    def forward(self, x):
        feature = self.bn(x)
        cls_score = self.classifier(feature)
        return feature, cls_score
class TFBNClassifier(nn.Module):
    '''bn + fc'''

    def __init__(self, in_dim, class_num):
        super(TFBNClassifier, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)

        self.bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feature = self.bn(x)
        cls_score = self.classifier(feature)
        return cls_score

def GeM(x):
    b, c, h, w = x.shape
    x = x.view(b, c, -1)
    p = 1.0
    x_pool = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)
    return x_pool

# class LFGA(nn.Module):
#     def __init__(self, id_dim, arti_dim, reduc_ratio=1):
#         super(LFGA, self).__init__()
#         self.high_dim = id_dim
#         self.low_dim = arti_dim
#         self.reduc_ratio = reduc_ratio
#
#         self.g = nn.Conv2d(self.low_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)
#         self.theta = nn.Conv2d(self.high_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)
#         self.phi = nn.Conv2d(self.low_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)
#
#         self.W = nn.Sequential(nn.Conv2d(self.low_dim//self.reduc_ratio, self.high_dim, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(arti_dim),)
#         nn.init.constant_(self.W[1].weight, 0.0)
#         nn.init.constant_(self.W[1].bias, 0.0)
#
#     def forward(self, x_h, x_l):
#         # B, C, H, W = x_h.size()
#         B = x_h.size(0) #128
#         g_x = self.g(x_l).view(B, self.low_dim, -1) #128,512,49
#         g_x = g_x.permute(0, 2, 1) #128,49,512
#
#         theta_x = self.theta(x_h).view(B, self.low_dim, -1) #128,512,49
#         theta_x = theta_x.permute(0, 2, 1) #128,49,512
#
#         phi_x = self.phi(x_l).view(B, self.low_dim, -1) #128,512,49
#
#         energy = torch.matmul(theta_x, phi_x) #128,49,49
#         attention = energy / energy.size(-1) #128,49,49
#
#         y = torch.matmul(attention, g_x) #128,49,512
#         y = y.permute(0, 2, 1).contiguous() #128,512,49
#         y = y.view(B, self.low_dim//self.reduc_ratio, *x_h.size()[2:]) #128,512,7,7
#         W_y = self.W(y)
#         z = W_y + x_h
#         return z


class CNL(nn.Module):
    def __init__(self, high_dim, low_dim, flag=0):
        super(CNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim

        self.g = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        if flag == 0:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(high_dim),)
        else:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=2, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=2, padding=0), nn.BatchNorm2d(self.high_dim), )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        phi_x = self.phi(x_l).view(B, self.low_dim, -1).permute(0, 2, 1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.view(B, self.low_dim, *x_l.size()[2:])
        W_y = self.W(y)
        # z = W_y + x_h
        z = W_y + x_l
        return z

class PNL(nn.Module):
    def __init__(self, high_dim, low_dim, reduc_ratio=1):
        super(PNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.reduc_ratio = reduc_ratio

        self.g = nn.Conv2d(self.low_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.low_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(nn.Conv2d(self.low_dim//self.reduc_ratio, self.high_dim, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(high_dim))
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x_l).view(B, self.low_dim, -1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.low_dim//self.reduc_ratio, *x_h.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h
        return z


class MFA_block(nn.Module):
    def __init__(self, high_dim, low_dim, flag):
        super(MFA_block, self).__init__()

        self.CNL = CNL(high_dim, low_dim, flag)
        # self.PNL = PNL(high_dim, low_dim)
    def forward(self, x, x0):
        z = self.CNL(x, x0)
        # z = self.PNL(z, x0)
        return z

class LFGA(nn.Module):
    def __init__(self, in_channel=3, ratio=4):
        super(LFGA, self).__init__()
        # self.gamma = nn.Parameter(torch.zeros(1))
        self.chanel_in = in_channel

        self.query_conv = nn.Conv2d(
            in_channels=in_channel, out_channels=in_channel//ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_channel, out_channels=in_channel//ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_channel, out_channels=in_channel, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(self.chanel_in)


    def forward(self, fa, fb):
        B, C, H, W = fa.size()
        proj_query = self.query_conv(fb).view(
            B, -1, H*W).permute(0, 2, 1)  # B , HW, C
        proj_key = self.key_conv(fb).view(
            B, -1, H*W)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # B, HW, HW
        attention = self.softmax(energy)  # BX (N) X (N)
        # attention = F.normalize(energy, dim=-1)

        proj_value = self.value_conv(fa).view(
            B, -1, H*W)  # B , C , HW

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        out = self.gamma*out + fa

        return self.relu(out)

class DEE_module(nn.Module):
    def  __init__(self, channel, reduction=16):
        super(DEE_module, self).__init__()

        self.FC11 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC11.apply(weights_init_kaiming)
        self.FC12 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC12.apply(weights_init_kaiming)
        self.FC13 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC13.apply(weights_init_kaiming)
        self.FC1 = nn.Conv2d(channel//4, channel, kernel_size=1)
        self.FC1.apply(weights_init_kaiming)

        self.FC21 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC21.apply(weights_init_kaiming)
        self.FC22 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC22.apply(weights_init_kaiming)
        self.FC23 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC23.apply(weights_init_kaiming)
        self.FC2 = nn.Conv2d(channel//4, channel, kernel_size=1)
        self.FC2.apply(weights_init_kaiming)
        self.dropout = nn.Dropout(p=0.01)

    def forward(self, x):
        x1 = (self.FC11(x) + self.FC12(x) + self.FC13(x))/3
        x1 = self.FC1(F.relu(x1))
        x2 = (self.FC21(x) + self.FC22(x) + self.FC23(x))/3
        x2 = self.FC2(F.relu(x2))
        out = torch.cat((x, x1, x2), 0)
        out = self.dropout(out)
        return out