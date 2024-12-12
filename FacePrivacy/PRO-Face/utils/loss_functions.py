import torch
import torch.nn as nn
import torchvision
import math
from torch.nn import TripletMarginWithDistanceLoss, CosineSimilarity
from pytorch_msssim import ssim
import lpips


class LPIPSLoss(nn.Module):
    """
    Part of pre-trained VGG16. This is used in case we want perceptual loss instead of Mean Square Error loss.
    See for instance https://arxiv.org/abs/1603.08155
    """
    def __init__(self, net='vgg'): # net can be 'alex' or 'vgg'
        super(LPIPSLoss, self).__init__()
        self.loss_func = lpips.LPIPS(net=net)

    def forward(self, x, y):
        _loss = self.loss_func(x, y).mean()
        return _loss


class VGGLoss(nn.Module):
    """
    Part of pre-trained VGG16. This is used in case we want perceptual loss instead of Mean Square Error loss.
    See for instance https://arxiv.org/abs/1603.08155
    """
    def __init__(self, block_no: int, layer_within_block: int, use_batch_norm_vgg: bool):
        super(VGGLoss, self).__init__()
        if use_batch_norm_vgg:
            vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        else:
            vgg16 = torchvision.models.vgg16(pretrained=True)
        curr_block = 1
        curr_layer = 1
        layers = []
        for layer in vgg16.features.children():
            layers.append(layer)
            if curr_block == block_no and curr_layer == layer_within_block:
                break
            if isinstance(layer, nn.MaxPool2d):
                curr_block += 1
                curr_layer = 1
            else:
                curr_layer += 1

        self.vgg = nn.Sequential(*layers)
        self.vgg.eval()

    def forward(self, x, y):
        mse_loss = nn.MSELoss()
        vgg_x = self.vgg(x)
        vgg_y = self.vgg(y)
        vgg_loss = mse_loss(vgg_x, vgg_y)
        return vgg_loss


class SSIMLoss(nn.Module):
    def __init__(self, data_range=(-1, 1)):
        super(SSIMLoss, self).__init__()
        self.data_range = data_range

    def normalize(self, x):
        data_min, data_max = self.data_range
        x_norm = (x - data_min) / (data_max - data_min)
        return x_norm

    def forward(self, x, y):
        x_norm = self.normalize(x)
        y_norm = self.normalize(y)
        ssim_loss = 1 - ssim(x_norm, y_norm, data_range=1, nonnegative_ssim=True)
        return ssim_loss


#### Define loss and evaluation functions
logits_loss = torch.nn.CrossEntropyLoss()
vgg_loss = VGGLoss(3, 1, False)
l1_loss = torch.nn.L1Loss()
triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
# triplet_loss = TripletMarginWithDistanceLoss(
#     distance_function=lambda x1, x2 : torch.acos(CosineSimilarity()(x1, x2)) / torch.tensor(math.pi), margin=1.0)
lpips_loss = LPIPSLoss()
# perceptual_triplet_loss = TripletMarginWithDistanceLoss(distance_function=lambda x, y : lpips_loss(x, y), margin=1.0)
