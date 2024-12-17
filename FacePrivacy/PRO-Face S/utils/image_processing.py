import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from FaceShifter.network.AEI_Net import AEI_Net
from FaceShifter.face_modules.model import Backbone
from SimSwap.options.test_options import TestOptions
from SimSwap.models.models import create_model
import config.config as c

import random
from PIL import Image
import numpy as np
import kornia


input_trans = transforms.Compose([
    transforms.Resize(256, interpolation=F.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])


input_trans_nonface = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5),
    transforms.Resize(160, interpolation=F.InterpolationMode.BICUBIC),
    transforms.RandomCrop(112)
])


def normalize(x: torch.Tensor, adaptive=False):
    _min, _max = -1, 1
    if adaptive:
        _min, _max = x.min(), x.max()
    x_norm = (x - _min) / (_max - _min)
    return x_norm


def clamp_normalize(x: torch.Tensor, lmin=-1, lmax=1):
    x_clamp = torch.clamp(x, lmin, lmax)
    x_norm = (x_clamp - lmin) / (lmax - lmin)
    return x_norm


class Blur(torch.nn.Module):
    def __init__(self, kernel_size, sigma_min, sigma_max):
        super().__init__()
        # self.random = True
        self.kernel_size = kernel_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_avg = (sigma_min + sigma_max) / 2

    def forward(self, img, *_):
        sigma = random.uniform(self.sigma_min, self.sigma_max) if self.training else self.sigma_avg
        img_blurred = F.gaussian_blur(img, self.kernel_size, [sigma, sigma])
        return img_blurred


class Pixelate(torch.nn.Module):
    def __init__(self, block_size_avg):
        super().__init__()
        if not isinstance(block_size_avg, int):
            raise ValueError("block_size_avg must be int")
        # self.random = True
        self.block_size_avg = block_size_avg
        self.block_size_list = range(block_size_avg -6, block_size_avg + 7, 2)

    def forward(self, img, *_):
        img_size = img.shape[-1]
        block_size = random.sample(self.block_size_list, 1)[0] if self.training else self.block_size_avg
        pixelated_size = img_size // block_size
        img_pixelated = F.resize(F.resize(img, pixelated_size), img_size, F.InterpolationMode.NEAREST)
        return img_pixelated


class MedianBlur(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        # self.random = True
        self.kernel = kernel_size
        self.size_min = kernel_size - 6
        self.size_max = kernel_size + 6

    def forward(self, img, *_):
        kernel_size = random.randint(self.size_min, self.size_max) if self.training else self.kernel
        if kernel_size % 2 == 0:
            kernel_size -= 1
        img_blurred = kornia.filters.median_blur(img, (kernel_size, kernel_size))
        return img_blurred





class SimSwap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        opt = TestOptions().parse()
        self.swapper = create_model(opt)
        self.swapper.eval()
        self.target_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.target_trans_inv = transforms.Compose([
            transforms.Normalize([0, 0, 0], [1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
        ])

    def forward(self, x, target_image: Image, device):
        # First convert PIL image to tensor
        target_image_tensor = self.target_trans(target_image).repeat(x.shape[0], 1, 1, 1).to(device)
        x_resize = F.resize(x.mul(0.5).add(0.5), [224, 224], F.InterpolationMode.BICUBIC)
        target_image_resize = F.resize(target_image_tensor, size=[112, 112])
        latend_id = self.swapper.netArc(target_image_resize)
        latend_id = latend_id.detach().to('cpu')
        latend_id = latend_id / np.linalg.norm(latend_id, axis=1, keepdims=True)
        latend_id = latend_id.to(device)
        x_swap = self.swapper(target_image_tensor, x_resize, latend_id, latend_id, True)
        latend_id.detach()
        target_image_resize.detach()
        x_resize.detach()
        x_swap = F.resize(x_swap.mul(2.0).sub(1.0), [256, 256], F.InterpolationMode.BICUBIC)
        target_image_tensor.detach()
        return x_swap


class FaceShifter(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.target_trans = transforms.Compose([
            transforms.Resize(112, interpolation=F.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])
        self.device = device
        self.G = AEI_Net(c_id=512)
        self.G.eval()

        self.G.load_state_dict(torch.load(c.DIR_FACESHIFTER_G_LATEST, map_location=device))
        self.G = self.G.to(device)
        self.arcface = Backbone(50, 0.6, 'ir_se').to(device)
        self.arcface.eval()
        self.arcface.load_state_dict(torch.load(c.DIR_FACESHIFTER_IRSE50, map_location=device),
                                strict=False)

    def forward(self, x, target_image: Image, *_):
        bs, _, w, h = x.shape
        target_image_tensor = self.target_trans(target_image).repeat(bs, 1, 1, 1).to(self.device)
        with torch.no_grad():
            embeds = self.arcface(F.resize(target_image_tensor, [112, 112], F.InterpolationMode.BILINEAR))
            yt, _ = self.G(F.resize(x, [256, 256], F.InterpolationMode.BILINEAR), embeds)
            target_image_tensor.detach()
            return yt


# RGBA image loader
def rgba_image_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGBA')


class Mask(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.target_trans = transforms.Compose([
            transforms.ToTensor(),
        ])

    def forward(self, img: torch.Tensor, overlay_img: Image, device):
        """
            Apply image masking by overlay an image with alpha channel on top of the a base image
            """
        # Get the alpha map
        overlay_tensor = self.target_trans(overlay_img).repeat(img.shape[0], 1, 1, 1).to(device)
        overlay_alpha = overlay_tensor[:, 3, :, :]

        # Compute non-zero region in the alpha map
        overlay_alpha_nonzero = torch.nonzero(overlay_alpha)
        (_, row_min, col_min), _ = overlay_alpha_nonzero.min(dim=0)
        (_, row_max, col_max), _ = overlay_alpha_nonzero.max(dim=0)

        # The followings computes the crop region that can keep the aspect ratio of the image
        height, width = int(row_max - row_min), int(col_max - col_min)
        center_y, center_x = int(row_min) + height // 2, int(col_min) + width // 2
        height = width = max(height, width)
        top, left = center_y - height // 2, center_x - height // 2
        img_size = img.shape[2]
        overlay_img_crop = F.resized_crop(overlay_tensor, top, left, height, width, [img_size, img_size])

        # Get the overlay image content and mask
        overlay_content = overlay_img_crop[:, :3, :, :].mul(2).sub(1)
        overlay_mask = overlay_img_crop[:, 3, :, :].unsqueeze(dim=1)

        # Apply the overlay
        img_masked = img * (1 - overlay_mask) + overlay_content * overlay_mask
        overlay_tensor.detach()
        return img_masked


class Obfuscator(torch.nn.Module):
    def __init__(self, options, device):
        super().__init__()
        self.name, *obf_params = options.split('_')
        # self.random = True
        self.fullname = options
        self.params = {}
        self.func = None
        self.device = device
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
        elif self.name == 'medianblur':
            kernel_size, = obf_params
            self.params['kernel_size'] = int(kernel_size)
            self.func = MedianBlur(self.params['kernel_size'])
        elif self.name == 'faceshifter':
            self.func = FaceShifter(self.device)
        elif self.name == 'simswap':
            self.func = SimSwap()
        elif self.name == 'mask':
            self.func = Mask()
        elif self.name == 'hybrid':
            self.functions = [Blur(21, 6, 10), MedianBlur(15), Pixelate(9)]
        elif self.name == 'hybridMorph':
            self.functions = [FaceShifter(self.device), SimSwap()]
        elif self.name == 'hybridAll':
            self.functions = [Blur(61, 9, 21), MedianBlur(23), Pixelate(20), FaceShifter(self.device), SimSwap(),
                              Mask()]

    def to(self, device):
        super(Obfuscator, self).to(device)
        self.device = device
        return self

    def forward(self, x, target=None):
        self.func.training = self.training
        return self.func(x, target, self.device)



