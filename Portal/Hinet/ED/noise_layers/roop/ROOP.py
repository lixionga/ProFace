import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import os
import torch.nn as nn
import random, string
import warnings
warnings.filterwarnings("ignore")

import insightface
import argparse
from network.noise_layers.mobilefaceswap.models.model import FaceSwap, l2_norm
from network.noise_layers.mobilefaceswap.models.arcface import IRBlock, ResNet
from network.noise_layers.mobilefaceswap.utils.align_face import back_matrix, dealign, align_img
from network.noise_layers.mobilefaceswap.utils.util import paddle2cv, cv2paddle
from network.noise_layers.mobilefaceswap.utils.prepare_data import LandmarkModel


class ROOP(nn.Module):
    def __init__(self, temp="temp/", target="/home/likaide/sda4/wxs/Dataset/dual_watermark/celeba_256/val/"):
        super(ROOP, self).__init__()
        self.temp = temp
        self.target = target
        if not os.path.exists(temp): os.mkdir(temp)
        model_path = 'network/noise_layers/roop/inswapper_128.onnx'
        self.model = insightface.model_zoo.get_model(model_path)
        self.model2 = insightface.app.FaceAnalysis(name='buffalo_l')
        self.model2.prepare(ctx_id=0, det_size=(320, 320))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


    def get_temp(self):
        return self.temp + ''.join(random.sample(string.ascii_letters + string.digits, 16)) + ".png" # png or jpg?

    def get_target(self):
        idx = random.randint(162771, 182637)
        return self.target + str(idx).zfill(6) + ".png"

    def forward(self, image_cover_mask):
        image, cover_image = image_cover_mask[0], image_cover_mask[1]

        noised_image = torch.zeros_like(image)

        for i in range(image.shape[0]):
            single_image = ((image[i].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).add(0.5).clamp(0, 255).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(single_image)

            file = self.get_temp()
            while os.path.exists(file):
                file = self.get_temp()

            try:
                self.swap_face(self.get_target(), im, file)
                fake = np.array(Image.open(file), dtype=np.uint8)
                os.remove(file)

                noised_image[i] = self.transform(fake).unsqueeze(0).to(image.device)
            except:
                print("Error in roop")
                noised_image[i] = image[i]

        return noised_image


    def swap_face(self, face_image_path, body_image, output_path):
        face_image = self.model2.get(cv2.imread(face_image_path))
        body_image = np.array(body_image)[:, :, ::-1]
        target_face = self.model2.get(body_image)
        result = self.model.get(body_image, target_face[0], face_image[0], paste_back=True)
        cv2.imwrite(output_path, result)
