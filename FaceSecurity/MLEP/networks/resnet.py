import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np
import torch

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False):
        super(ResNet, self).__init__()

        self.unfoldSize = 2
        self.unfoldIndex = 0
        assert self.unfoldSize > 1
        assert -1 < self.unfoldIndex and self.unfoldIndex < self.unfoldSize*self.unfoldSize
        self.inplanes = 64
        self.conv1 = nn.Conv2d(9, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 , layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc1 = nn.Linear(512 * block.expansion, 1)
        self.fc1 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    # patch-based entropy
    def calculate_information_entropy(self, x):
        """
        Independently calculate local color entropy for each channel of the input image x.
        The local block size is 2x2, allowing for overlapping calculations.
        Color entropy is assigned based on the following five situations:
        1. All pixels are the same: entropy value is 0.0
        2. Three pixels are the same, one is different: entropy value is 0.81
        3. Two pairs of pixels are the same: entropy value of 1.0
        4. Two pixels are the same, the other two are different: entropy value is 1.5
        5. All pixels are different: entropy value is 2.0
        """

        batch, channels, H, W = x.size()

        # Extract 2x2 sliding windows with a stride of 1
        patches = x.unfold(2, 2, 1).unfold(3, 2, 1)  # (batch, channels, H-1, W-1, 2, 2)
        # print("patches shape: ",patches.shape)
        # reshaping as (batch, channels, (H-1)*(W-1), 4)
        patches = patches.contiguous().view(batch, channels, (H-1)*(W-1), 4)
        # print("After contiguous patches shape: ",patches.shape)
        # Extract 4 pixel values from each 2x2 block
        p0 = patches[:, :, :, 0]
        p1 = patches[:, :, :, 1]
        p2 = patches[:, :, :, 2]
        p3 = patches[:, :, :, 3]
        # print("p0 shape: ", p0.shape)
        # Calculate masks for various situations
        all_same = (p0 == p1) & (p1 == p2) & (p2 == p3)  # All four pixel values are the same.
        # print("all_same shape: ",all_same.shape)
        
        three_same = ((p0 == p1) & (p1 == p2) & (p2 != p3)) | \
                     ((p0 == p1) & (p1 == p3) & (p2 != p3)) | \
                     ((p0 == p2) & (p2 == p3) & (p1 != p2)) | \
                     ((p1 == p2) & (p2 == p3) & (p0 != p1))  # Three pixels are the same, one is different
        
        
        two_pairs = ((p0 == p1) & (p2 == p3) & (p1 != p2)) | \
                    ((p0 == p2) & (p1 == p3) & (p0 != p1)) | \
                    ((p0 == p3) & (p1 == p2) & (p0 != p1))  # Two pairs of pixels are identical.
        
        two_same_two_diff = ((p0 == p1) & (p2 != p3) & (p1 != p2) & (p1 != p3)) | \
                            ((p0 == p2) & (p1 != p3) & (p0 != p1) & (p0 != p3)) | \
                            ((p0 == p3) & (p1 != p2) & (p0 != p1) & (p0 != p2)) | \
                            ((p1 == p2) & (p0 != p3) & (p1 != p0) & (p1 != p3)) | \
                            ((p1 == p3) & (p0 != p2) & (p1 != p0) & (p1 != p2)) | \
                            ((p2 == p3) & (p0 != p1) & (p2 != p0) & (p2 != p1))  # Two pixels are the same, the other two are different.
        
        all_diff = (p0 != p1) & (p0 != p2) & (p0 != p3) & \
                   (p1 != p2) & (p1 != p3) & (p2 != p3)  # All pixels are different.
        
        # Initialize category index.
        entropy_image = torch.zeros_like(p0, dtype=torch.float, device=x.device)
        entropy_image = torch.where(all_same, torch.tensor(0.0, device=x.device), entropy_image)
        # entropy_image = torch.where(three_same, torch.tensor(0.81, device=x.device), entropy_image)
        entropy_image = torch.where(three_same, torch.tensor(0.8, device=x.device), entropy_image)
        entropy_image = torch.where(two_pairs, torch.tensor(1.0, device=x.device), entropy_image)
        entropy_image = torch.where(two_same_two_diff, torch.tensor(1.5, device=x.device), entropy_image)
        entropy_image = torch.where(all_diff, torch.tensor(2.0, device=x.device), entropy_image)
        
        entropy_image = entropy_image.view(batch, channels, H-1, W-1)

        return entropy_image

    # Divide and shuffle.
    def random_rearrange_blocks(self, x, block_size=2):
        """
        Split the input image x into 8x8 small blocks and arrange them randomly.
        If the image size cannot be divided by 8, the excess part will be cropped.

        parameter:
            x (torch.Tensor): input image，with the shape of (batch, channels, H, W).
            block_size (int): size of blocks，the default is 8.

        return:
            rearranged_x (torch.Tensor): The randomly arranged image, with the shape of (batch, channels, H', W').
        """
        batch, channels, H, W = x.size()

        # Crop the image so that its size can be evenly divided by block_size.
        H_cropped = (H // block_size) * block_size
        W_cropped = (W // block_size) * block_size
        x_cropped = x[:, :, :H_cropped, :W_cropped]

        # Split the image into small blocks.
        blocks = x_cropped.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
        num_blocks_H = blocks.size(2)
        num_blocks_W = blocks.size(3)
        
        # Reshape as (batch, channels, num_blocks, block_size, block_size).
        blocks = blocks.contiguous().view(batch, channels, num_blocks_H * num_blocks_W, block_size, block_size)
        
        # Fixed the seed (Attention, please annotate this line during training *).
        torch.manual_seed(99)

        # Randomly arrange blocks for each image
        for i in range(batch):
            permuted_indices = torch.randperm(num_blocks_H * num_blocks_W)
            blocks[i] = blocks[i, :, permuted_indices, :, :]

        # Reorder the blocks into (batch, channels, num_blocks_H, num_blocks_W, block_size, block_size)
        blocks = blocks.view(batch, channels, num_blocks_H, num_blocks_W, block_size, block_size)

        # Splicing blocks to reconstruct images
        rearranged_x = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        rearranged_x = rearranged_x.view(batch, channels, H_cropped, W_cropped)

        return rearranged_x
    
    # def save_intermediate_image(self, x, folder):
    #     """
    #     Save the image to the specified folder.
    #     """
    #     # Normalize to [0, 1] range if necessary
    #     x = x.clamp(0, 2.0) / 2.0  # Ensure the values are in range [0, 1]

    #     # Convert from tensor to image and save
    #     save_dir = Path(folder)
    #     save_dir.mkdir(parents=True, exist_ok=True)

    #     # Iterate through the batch dimension if it's > 1
    #     for i in range(x.size(0)):
    #         # Get the i-th image from the batch (assuming 3 channels)
    #         img = x[i]

    #         # Convert tensor to PIL image and save it
    #         img_name = f"image_{i}.png"  # You can customize the name
    #         img_path = save_dir / img_name

    #         # Save the image
    #         save_image(img, img_path)

    #         print(f"Saved intermediate image to {img_path}")

    def forward(self, x):

        # ---------MIE start here---------
        # print(x.shape)
        # Divide and shuffle
        x_rearrange = self.random_rearrange_blocks(x, 2)

        # Bilinear downsampling with a scale factor of 0.5 and upsampling.
        x_rearrange_down_2 = F.interpolate(x_rearrange, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_rearrange_up_2 = F.interpolate(x_rearrange_down_2, size=x_rearrange.shape[2:], mode='bilinear', align_corners=False)

        # Bilinear downsampling with a scale factor of 0.25 and upsampling.
        x_rearrange_down_4 = F.interpolate(x_rearrange, scale_factor=0.25, mode='bilinear', align_corners=False)
        x_rearrange_up_4 = F.interpolate(x_rearrange_down_4, size=x_rearrange.shape[2:], mode='bilinear', align_corners=False)

        # Calculate information entropy feature.
        entropy_image = self.calculate_information_entropy(x_rearrange)
        entropy_image_2 = self.calculate_information_entropy(x_rearrange_up_2)
        entropy_image_4 = self.calculate_information_entropy(x_rearrange_up_4)

        # Concat the three.
        concat_result = torch.cat([entropy_image, entropy_image_2, entropy_image_4], dim=1)

        # Perform subsequent convolution. (On line 112, we changed the input channel of conv1 to 9.)
        x = self.conv1(concat_result)

        # ---------MIE end here---------

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        checkpoint = torch.load('/home/lxw/code/mine/MLEP-Detection/checkpoints/MLEP_resnet502025_01_14_09_05_49/model_epoch_180.pth')
        model.load_state_dict(checkpoint)
        # model = resnet50(weight_path='/home/lxw/code/mine/MLEP-Detection/checkpoints/MLEP_resnet502025_01_14_09_05_49/model_epoch_180.pth')
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
