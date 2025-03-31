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

import paddle
import argparse
from network.noise_layers.mobilefaceswap.models.model import FaceSwap, l2_norm
from network.noise_layers.mobilefaceswap.models.arcface import IRBlock, ResNet
from network.noise_layers.mobilefaceswap.utils.align_face import back_matrix, dealign, align_img
from network.noise_layers.mobilefaceswap.utils.util import paddle2cv, cv2paddle
from network.noise_layers.mobilefaceswap.utils.prepare_data import LandmarkModel


class MobileFaceSwap(nn.Module):
    def __init__(self, temp="temp/", target="/home/likaide/sda4/wxs/Dataset/dual_watermark/celeba_256/val/"):
        super(MobileFaceSwap, self).__init__()
        self.temp = temp
        self.target = target
        if not os.path.exists(temp): os.mkdir(temp)

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
                self.image_test(self.get_target(), im, file)
                fake = np.array(Image.open(file), dtype=np.uint8)
                os.remove(file)

                noised_image[i] = self.transform(fake).unsqueeze(0).to(image.device)
            except:
                print("Error in mobilefaceswap")
                noised_image[i] = image[i]

        return noised_image


    def get_id_emb(self, id_net, id_img_path):
        id_img = cv2.imread(id_img_path)

        id_img = cv2.resize(id_img, (112, 112))
        id_img = cv2paddle(id_img)
        mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
        std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
        id_img = (id_img - mean) / std

        id_emb, id_feature = id_net(id_img)
        id_emb = l2_norm(id_emb)

        return id_emb, id_feature


    def image_test(self, face_image_path, body_image, output_path):
        use_gpu = False
        paddle.set_device("gpu" if use_gpu else 'cpu')
        faceswap_model = FaceSwap(use_gpu)

        id_net = ResNet(block=IRBlock, layers=[3, 4, 23, 3])
        id_net.set_dict(paddle.load('network/noise_layers/mobilefaceswap/checkpoints/arcface.pdparams'))

        id_net.eval()

        weight = paddle.load('network/noise_layers/mobilefaceswap/checkpoints/MobileFaceSwap_224.pdparams')

        base_path = face_image_path  # .replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        id_emb, id_feature = self.get_id_emb(id_net, base_path)  # + '_aligned.png')

        faceswap_model.set_model_param(id_emb, id_feature, model_weight=weight)
        faceswap_model.eval()

        att_img = np.array(body_image)[:, :, ::-1]
        att_img = cv2paddle(att_img)

        res, mask = faceswap_model(att_img)
        res = paddle2cv(res)

        cv2.imwrite(output_path, res)

