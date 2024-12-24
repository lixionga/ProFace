import os.path
import sys
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import random
from PIL import Image
import numpy as np
import kornia
from SimSwap.options.test_options import TestOptions
from SimSwap.models.models import create_model
from config import config as c


input_trans = transforms.Compose([
    transforms.Resize(112, interpolation=F.InterpolationMode.BICUBIC),
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


def adaptive_normalize(x: torch.Tensor):
    y = (x - x.mean()) / x.std()
    return normalize(y)

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
        self.block_size_min = block_size_avg - 4
        self.block_size_max = block_size_avg + 4

    def forward(self, img, *_):
        img_size = img.shape[-1]
        block_size = random.randint(self.block_size_min, self.block_size_max) if self.training else self.block_size_avg
        pixelated_size = img_size // block_size
        img_pixelated = F.resize(F.resize(img, pixelated_size), img_size, F.InterpolationMode.NEAREST)
        return img_pixelated


class MedianBlur(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        # self.random = True
        self.kernel = kernel_size
        self.size_min = kernel_size - 7
        self.size_max = kernel_size + 7

    def forward(self, img, *_):
        kernel_size = random.randint(self.size_min, self.size_max) if self.training else self.kernel
        if kernel_size % 2 == 0:
            kernel_size -= 1
        img_blurred = kornia.filters.median_blur(img, (kernel_size, kernel_size))
        return img_blurred


# target_trans_faceshifter = transforms.Compose([
#     transforms.Resize(112, interpolation=F.InterpolationMode.BICUBIC),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=0.5, std=0.5)
# ])
#
# target_trans_simswap = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])


class SimSwap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        opt = TestOptions().parse()
        opt.Arc_path = os.path.join(c.DIR_PROJECT, 'SimSwap/arcface_model/arcface_checkpoint.tar')
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

    def forward(self, x, noise_mk: Image, device):
        # First convert PIL image to tensor
        # target_image_tensor = target_image
        x_resize = F.resize(x.mul(0.5).add(0.5), [224, 224], F.InterpolationMode.BICUBIC)
        # target_image_resize = F.resize(target_image_tensor, size=[112, 112])
        latend_id = self.swapper.netArc(x)
        latend_id =noise_mk(latend_id)
        latend_id = latend_id.detach().to('cpu')
        latend_id = latend_id / np.linalg.norm(latend_id, axis=1, keepdims=True)
        latend_id = latend_id.to(device)
        x_swap = self.swapper(None, x_resize, latend_id, latend_id, True)
        latend_id.detach()
        # target_image_resize.detach()
        x_resize.detach()
        x_swap = F.resize(x_swap.mul(2.0).sub(1.0), [112, 112], F.InterpolationMode.BICUBIC)
        # target_image_tensor.detach()
        return x_swap




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
        elif self.name == 'simswap':
            self.func = SimSwap()
            # self.targ_img_trans = target_trans_simswap
            # self.targ_img_trans_inv = transforms.Compose([
            #     transforms.Normalize([0, 0, 0], [1 / 0.229, 1 / 0.224, 1 / 0.225]),
            #     transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
            # ])
        elif self.name == 'mask':
            self.func = Mask()
        elif self.name == 'hybrid':
            self.functions = [Blur(21, 6, 10), MedianBlur(15), Pixelate(9)]

    def to(self, device):
        super(Obfuscator, self).to(device)
        self.device = device
        return self

    def forward(self, x, target=None):
        # obf_func = self.func
        # if self.name.startswith('hybrid'):
        #     obf_func = random.choice(self.functions)
        self.func.training = self.training
        return self.func(x, target, self.device)

    # def proc(self, x, target=None, obf_type='blur'):
    #     obf_func = self.functions2[obf_type]
    #     obf_func.training = self.training
    #     return obf_func(x, target)


def test_obfuscator():
    from PIL import Image
    from torchvision.utils import save_image
    img = Image.open('images/lfw_112_sample.jpg')
    trans = transforms.Compose([
        transforms.Resize(112, interpolation=F.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    img_tensor = trans(img)
    img_tensor_batch = img_tensor.repeat(8, 1, 1, 1)

    ## Test blur
    # import numpy as np
    # sigma = np.arange(2, 7, 0.5)
    # kernel = [11]
    blur = Obfuscator('blur', 11, 2, 6)
    print(blur.name)
    print(blur.params)
    print(blur.fullname)
    for i in range(5):
        blur.train()
        img_tensor_batch_blurred = blur(img_tensor_batch)
        save_image(img_tensor_batch_blurred, f'images/test_blur_random_{i}.jpg')
    for i in range(5):
        blur.eval()
        img_tensor_batch_blurred = blur(img_tensor_batch)
        save_image(img_tensor_batch_blurred, f'images/test_blur_fixed_{i}.jpg')

    ## Test pixelation
    pixelate = Obfuscator('pixelate', '4', '10')
    print(pixelate.name)
    print(pixelate.params)
    print(pixelate.fullname)
    for i in range(5):
        pixelate.train()
        img_tensor_batch_pixelated = pixelate(img_tensor_batch)
        save_image(img_tensor_batch_pixelated, f'images/test_pixelate_random_{i}.jpg')
    for i in range(5):
        pixelate.eval()
        img_tensor_batch_pixelated = pixelate(img_tensor_batch)
        save_image(img_tensor_batch_pixelated, f'images/test_pixelate_fixed_{i}.jpg')


def image_mask(img: torch.Tensor, overlay_img: torch.Tensor):
    """
    Apply image masking by overlay an image with alpha channel on top of the a base image
    """
    # Get the alpha map
    overlay_alpha = overlay_img[:, 3, :, :]

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
    overlay_img_crop = F.resized_crop(overlay_img, top, left, height, width, [img_size, img_size])

    # Get the overlay image content and mask
    overlay_content = overlay_img_crop[:, :3, :, :]
    overlay_mask = overlay_img_crop[:, 3, :, :].unsqueeze(dim=1)

    # Apply the overlay
    img_masked = img * (1 - overlay_mask) + overlay_content * overlay_mask
    return img_masked


def test_blur_pixelate():
    from torchvision.utils import save_image
    img = Image.open('images/original_crop.jpg')
    img_tensor = transforms.ToTensor()(img)
    # img_tensor_batch = img_tensor.repeat(8, 1, 1, 1)
    ## Test blur
    img_tensor_batch_blurred = image_blur(img_tensor, 31, 7.0)
    save_image(img_tensor_batch_blurred, 'images/test_blur.jpg')
    ## Test pixelation
    img_tensor_batch_pixelated = image_pixelate(img_tensor, 10)
    save_image(img_tensor_batch_pixelated, 'images/test_pixelate.jpg')


def test_image_overlay():
    from torchvision.utils import save_image
    img_cartoon = Image.open('images/cartoon_sample.png')
    img_face = Image.open('images/celeba_aligned_224_sample.jpg')
    cartoon_tensor = transforms.ToTensor()(img_cartoon).repeat(8, 1, 1, 1)
    face_tensor = transforms.ToTensor()(img_face).repeat(8, 1, 1, 1)
    composition = image_mask(face_tensor, cartoon_tensor)
    save_image(composition, 'images/composition.jpg')


def test_image_median_filter():
    from torchvision.utils import save_image
    img = Image.open('test/original_crop_112.jpg')
    img_tensor = transforms.ToTensor()(img)
    img_tensor_batch = img_tensor.repeat(8, 1, 1, 1)
    ## Test blur
    for i in [5, 7, 9, 11, 13, 15]:
        blur = MedianBlur(i)
        img_blur = blur(img_tensor_batch)
        save_image(img_blur, f'test/original_crop_median_blur_{i}.jpg')



if __name__ == '__main__':
    # test_blur_pixelate()
    # test_image_overlay()
    test_obfuscator()
    # test_blur_pixelate()
    # test_image_median_filter()
