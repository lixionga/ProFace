import glob
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch
import torch.utils.data as data


# __all__ = ['CASIA']

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class MyDataset(data.Dataset):

    def __init__(self):
        self.files = sorted(glob.glob('/Data/FFHQ256/image/test' + "/*.png"))

    def __getitem__(self, index):
        try:
            image_path = self.files[index]
            image = Image.open(image_path)

            return image
        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.files)
