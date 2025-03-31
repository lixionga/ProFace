import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
import config as c
from ESRGAN import Discriminator
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


def get_one_image(path):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image = Image.open(path)
    image = to_rgb(image)
    image = data_transform(image)
    return image.unsqueeze(0)


image_path = "/home/ysc/HiNet/image/diff_secret_mp/diff_mp_secret00000.jpg"

on_img = "/home/ysc/HiNet/image/secret-mp-rev/secret_mp_rev00000.jpg"

rgb_img = np.array(Image.open(on_img).convert('RGB')).astype(np.float32) / 255.0

model = Discriminator(input_shape=(c.channels_in, c.cropsize, c.cropsize)).to(c.device)
target_layers = [model.model[-2]]
input_tensor = get_one_image(image_path).to(c.device)  # Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers)

targets = [BinaryClassifierOutputTarget(0)]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# You can also get the model outputs without having to re-inference
# model_outputs = cam.outputs

cv2.imwrite("/home/ysc/HiNet/image/CAM/secret_mp_rev00000.jpg", visualization)
