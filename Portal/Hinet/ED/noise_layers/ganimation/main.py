import os
import torch
import torch.nn as nn
from PIL import Image
import random, string
from torchvision import transforms
import numpy as np
from torchvision.utils import save_image
from network.noise_layers.ganimation.model import Generator

class GANimation(nn.Module):
    def __init__(self, temp="temp/", target="/home/likaide/sda4/wxs/Dataset/dual_watermark/celeba_256/val/"):
        super(GANimation, self).__init__()
        self.temp = temp
        self.target = target
        self.G = Generator(64, 17, 6)
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

    def imFromAttReg(self, att, reg, x_real):
        """Mixes attention, color and real images"""
        return (1-att)*reg + att*x_real

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
                self.animation(self.get_target(), im, file)
                fake = np.array(Image.open(file), dtype=np.uint8)
                os.remove(file)
                noised_image[i] = self.transform(fake).unsqueeze(0).to(image.device)
            except:
                print("Error in ganimation")
                noised_image[i] = image[i]

        return noised_image

    def animation(self, face_image_path, body_image, output_path):
        regular_image_transform = []
        regular_image_transform.append(transforms.ToTensor())
        regular_image_transform.append(transforms.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        regular_image_transform = transforms.Compose(regular_image_transform)

        G_path = 'network/noise_layers/ganimation/eric_andre/pretrained_models/7001-37-G.ckpt'
        self.G.load_state_dict(torch.load(G_path, map_location=f'cuda:{0}'))
        self.G = self.G.cuda(0)

        reference_expression_images = []
        animation_attributes_path = 'network/noise_layers/ganimation/eric_andre/attributes.txt'
        animation_attribute_images_dir = 'network/noise_layers/ganimation/eric_andre/attribute_images'

        with torch.no_grad():
            with open(animation_attributes_path, 'r') as txt_file:
                image_to_animate = regular_image_transform(body_image).unsqueeze(0).cuda()

                csv_lines = txt_file.readlines()

                targets = torch.zeros(len(csv_lines), 17)
                input_images = torch.zeros(len(csv_lines), 3, 128, 128)

                for idx, line in enumerate(csv_lines):
                    splitted_lines = line.split(' ')
                    image_path = os.path.join(animation_attribute_images_dir, splitted_lines[0])
                    input_images[idx, :] = regular_image_transform(Image.open(image_path)).cuda()
                    reference_expression_images.append(splitted_lines[0])
                    targets[idx, :] = torch.Tensor(np.array(list(map(lambda x: float(x) / 5., splitted_lines[1::]))))

            #for target_idx in range(targets.size(0)):
            target_idx = random.randint(0, targets.size(0)-1)
            targets_au = targets[target_idx, :].unsqueeze(0).cuda()
            resulting_images_att, resulting_images_reg = self.G(image_to_animate, targets_au)
            resulting_image = self.imFromAttReg(resulting_images_att, resulting_images_reg, image_to_animate).cuda()

            save_image((resulting_image + 1) / 2, os.path.join(output_path))
