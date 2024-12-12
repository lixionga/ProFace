import os.path
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from FaceShifter.face_shifter import face_shifter
import random
import numpy as np
from SimSwap.options.test_options import TestOptions
from SimSwap.models.models import create_model
from config import config as c


input_trans = transforms.Compose([
    transforms.Resize(112, interpolation=F.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])


def normalize(x: torch.Tensor, lower=-1, upper=1, adaptive=False):
    _min, _max = lower, upper
    if adaptive:
        _min, _max = x.min(), x.max()
    x_norm = (x - _min) / (_max - _min)
    return x_norm


def image_blur(img: torch.Tensor, kernel_size=81, sigma=8.0):
    trans_blur = transforms.GaussianBlur(kernel_size, sigma)
    img_blurred = trans_blur(img)
    return img_blurred


def image_pixelate(img: torch.Tensor, block_size=10):
    img_size = img.shape[-1]
    pixelated_size = img_size // block_size
    trans_pixelate = transforms.Compose([
        transforms.Resize(pixelated_size),
        transforms.Resize(img_size, F.InterpolationMode.NEAREST),
    ])
    img_pixelated = trans_pixelate(img)
    return img_pixelated


class Blur(torch.nn.Module):
    def __init__(self, kernel_size, sigma_min, sigma_max):
        super().__init__()
        self.random = True
        self.kernel_size = kernel_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_avg = (sigma_min + sigma_max) / 2

    def forward(self, img):
        sigma = random.uniform(self.sigma_min, self.sigma_max) if self.random else self.sigma_avg
        img_blurred = F.gaussian_blur(img, self.kernel_size, [sigma, sigma])
        return img_blurred


class Pixelate(torch.nn.Module):
    def __init__(self, block_size_avg):
        super().__init__()
        if not isinstance(block_size_avg, int):
            raise ValueError("block_size_avg must be int")
        self.random = True
        self.block_size_avg = block_size_avg
        self.block_size_min = block_size_avg - 3
        self.block_size_max = block_size_avg + 3

    def forward(self, img):
        img_size = img.shape[-1]
        block_size = random.randint(self.block_size_min, self.block_size_max) if self.random else self.block_size_avg
        pixelated_size = img_size // block_size
        img_pixelated = F.resize(F.resize(img, pixelated_size), img_size, F.InterpolationMode.NEAREST)
        return img_pixelated


class SimSwap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        opt = TestOptions().parse()
        opt.Arc_path = os.path.join(c.PROJECT_DIR, 'SimSwap/arcface_model/arcface_checkpoint.tar')
        self.swapper = create_model(opt)
        self.swapper.eval()

    def forward(self, x, target_image):
        x_resize = F.resize(x.mul(0.5).add(0.5), [224, 224], F.InterpolationMode.BICUBIC)
        target_image_resize = F.resize(target_image, size=[112, 112])
        latend_id = self.swapper.netArc(target_image_resize)
        latend_id = latend_id.detach().to('cpu')
        latend_id = latend_id / np.linalg.norm(latend_id, axis=1, keepdims=True)
        latend_id = latend_id.to(c.device)
        x_swap = self.swapper(target_image, x_resize, latend_id, latend_id, True)
        latend_id.detach()
        target_image_resize.detach()
        x_resize.detach()
        x_swap = F.resize(x_swap.mul(2.0).sub(1.0), [112, 112], F.InterpolationMode.BICUBIC)
        return x_swap


class Obfuscator(torch.nn.Module):
    def __init__(self, options):
        super().__init__()
        self.name, *obf_params = options.split('_')
        self.random = True
        self.fullname = options
        self.params = {}
        self.func = None
        if self.name == 'blur':
            kernel_size, sigma_min, sigma_max = obf_params
            self.params['kernal_size'] = int(kernel_size)
            self.params['sigma_min'] = float(sigma_min)
            self.params['sigma_max'] = float(sigma_max)
            self.func = Blur(self.params['kernal_size'], self.params['sigma_min'], self.params['sigma_max'])
        elif self.name == 'pixelate':
            block_size_avg, = obf_params
            self.params['block_size_avg'] = int(block_size_avg)
            self.func = Pixelate(self.params['block_size_avg'])
        elif self.name == 'faceshifter':
            self.func = face_shifter
            self.targ_img_trans = transforms.Compose([
                transforms.Resize(112, interpolation=F.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5)
            ])
            self.targ_img_trans_inv = transforms.Compose([
                transforms.Normalize(mean=-1, std=2)
            ])
        elif self.name == 'simswap':
            self.func = SimSwap()
            self.targ_img_trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.targ_img_trans_inv = transforms.Compose([
                transforms.Normalize([0, 0, 0], [1 / 0.229, 1 / 0.224, 1 / 0.225]),
                transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
            ])

    def train(self, mode: bool = True):
        self.func.random = mode

    def eval(self):
        self.func.random = False

    def forward(self, x):
        return self.func(x)

    def swap(self, x, y):
        return self.func(x, y)
