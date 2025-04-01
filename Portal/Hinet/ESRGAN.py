import torch.nn as nn
import torch
from torchvision.models import resnet50
import torch.nn.functional as F
import Hinet.config as c


# patchDiscriminator 1.5G
class Discriminator(nn.Module):  # 分类器
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = c.discriminator_pred_dim
        self.feat_size = patch_h * patch_w

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

        # 定义全连接层
        self.fc = nn.Linear(patch_h * patch_w, self.output_shape)

    def forward(self, img):
        feature = self.model(img)
        out = torch.flatten(feature, start_dim=1)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out

# 残差块
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#         self.shortcut = nn.Sequential()
#         if in_channels != out_channels or stride != 1:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
#
#
# class EnsembleNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.block1 = ResidualBlock(2, 64, stride=1)
#         self.block2 = ResidualBlock(64, 128, stride=2)
#
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(128, 1)
#
#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         return torch.sigmoid(self.fc(x))


#
class Resnet50cls(nn.Module):  # 2.5G
    def __init__(self, dropout_prob=0.1):
        super().__init__()
        self.resnet50 = resnet50(pretrained=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 128)  # 输出层只有一个单元，用于二分类
        self.fc3 = nn.Linear(128, c.discriminator_pred_dim)  # 输出层只有一个单元，用于二分类

    def forward(self, inputs):
        features = self.resnet50(inputs)  # [batch_size, 1000]
        x = self.dropout(features)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)  # [batch_size, 1]
        return torch.sigmoid(x)
