# -*- coding:utf-8 -*-
import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise, KPCA_CAM
)
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from model import Detector

def generate_cam_image(device, image_path, aug_smooth, eigen_smooth, method, output_dir):
    """
    生成指定方法的类激活映射（CAM）图像并保存。

    参数:
    device (str): 使用的计算设备，如 'cpu' 或 'cuda' 等。
    image_path (str): 输入图像的路径。
    aug_smooth (bool): 是否应用测试时增强来平滑CAM。
    eigen_smooth (bool): 是否通过取cam_weights*activations的第一主成分来减少噪声。
    method (str): 要使用的CAM方法，可选值包括 'gradcam', 'hirescam' 等多种方法。
    output_dir (str): 保存生成图像的输出目录。
    """
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise,
        'kpcacam': KPCA_CAM
    }

    model = Detector().to(device).eval()
    model_path = 'weights/EG_FF++(raw).tar'
    model.load_state_dict(torch.load(model_path,map_location=torch.device(device)), strict=False)
    model = model.net.to(torch.device(device)).eval()

    target_layers = [model._blocks[-1]]
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]).to(device)

    targets = None

    cam_algorithm = methods[method]
    with cam_algorithm(model=model, target_layers=target_layers) as cam:
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=aug_smooth,
                            eigen_smooth=eigen_smooth)

        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    img_path=image_path
    # 获取输入图像的文件名（不含路径和扩展名），用于后续生成输出图像的文件名
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    # 构建输出CAM图像的完整路径，将根据方法名称和基础文件名来命名保存的图像文件，保存在指定的输出目录下
    cam_output_path = os.path.join(output_dir, f'{base_filename}_{method}_cam.jpg')
    cv2.imwrite(cam_output_path, cam_image)
    print(f'Saved CAM image to {output_dir}')