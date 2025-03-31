import torch
import numpy as np
import cv2

# 从 OpenCV (BGR) 图像转为 PyTorch 张量（不包括颜色空间转换）
def cvpaddle(img):
    # 转换为 (C, H, W) 格式
    img = np.transpose(img, (2, 0, 1))
    # 转为 PyTorch 张量，并添加一个批次维度
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    # 归一化到 [0, 1]
    img = img / 255.0
    return img

# 从 OpenCV (BGR) 图像转为 PyTorch 张量（包括颜色空间转换）
def cv2tensor(img):
    # 从 BGR 转为 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 转换为 (C, H, W) 格式
    img = np.transpose(img, (2, 0, 1))
    # 转为 PyTorch 张量，并添加一个批次维度
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    # 归一化到 [0, 1]
    img = img / 255.0
    return img

# 从 PyTorch 张量转为 OpenCV (BGR) 图像
def tesnor2cv(img):
    # 移除批次维度
    img = img.squeeze(0).detach().cpu().numpy()
    # 转换为 (H, W, C) 格式
    img = np.transpose(img, (1, 2, 0))
    # 反归一化到 [0, 255]
    img *= 255
    img = img.astype(np.uint8)
    # 从 RGB 转为 BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


