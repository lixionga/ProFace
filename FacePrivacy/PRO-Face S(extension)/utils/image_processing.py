import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from SimSwap.options.test_options import TestOptions
from SimSwap.models.models import create_model
import config.config as c
from models.model import Model, KeyEmbedding
import random
from PIL import Image
import numpy as np


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






class PreProcessing(torch.nn.Module):
    def __init__(self, device='cuda:0'):
        super().__init__()
        self.device = device

        self.net = Model()
        self._load_model(self.net, 'generator',
                         'generator_state_dict')

        self.keyembed = KeyEmbedding()
        self._load_model(self.keyembed, 'key_embedding',
                         'mlp_state_dict')

        self.model = SimSwap()
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, model, path, key):
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint[key])
        model.to(self.device)
        model.eval()



    def forward(self, x, key):
        transformer_Arcface = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_id = transformer_Arcface(x)
        img_id_downsample = torch.nn.functional.interpolate(img_id, size=(112, 112))
        latent_id = self.model.swapper.netArc(img_id_downsample)
        latent_id = torch.nn.functional.normalize(latent_id, p=2, dim=1)

        z_embedding = self.keyembed(key)
        with torch.no_grad():
            fake_id = self.net(torch.cat((latent_id, z_embedding), dim=1))
            fake_id = fake_id[:, :512]
            anonymized_face = self.model.swapper.netG(x, fake_id).detach()

        return anonymized_face





def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            # param.data = c.init_scale * torch.randn(param.data.shape).cuda()
            param.data = 0.01 * torch.randn(param.data.shape).cuda(1)  # 用的hi后修改为0
            if split[-2] == 'conv5':
                param.data.fill_(0.)






def unsigned_long_to_binary_repr(unsigned_long, passwd_length):
    batch_size = unsigned_long.shape[0]
    target_size = passwd_length // 4

    binary = np.empty((batch_size, passwd_length), dtype=np.float32)
    for idx in range(batch_size):
        binary[idx, :] = np.array([int(item) for item in bin(unsigned_long[idx])[2:].zfill(passwd_length)])

    dis_target = np.empty((batch_size, target_size), dtype=np.int64)
    for idx in range(batch_size):
        tmp = unsigned_long[idx]
        for byte_idx in range(target_size):
            dis_target[idx, target_size - 1 - byte_idx] = tmp % 16
            tmp //= 16
    return binary, dis_target


def generate_code(passwd_length, batch_size, device, inv, use_minus_one, gen_random_WR):
    unsigned_long = np.random.randint(0, 2 ** passwd_length, size=(batch_size,), dtype=np.uint64)
    binary, dis_target = unsigned_long_to_binary_repr(unsigned_long, passwd_length)
    z = torch.from_numpy(binary).to(device)
    dis_target = torch.from_numpy(dis_target).to(device)

    repeated = True
    while repeated:
        rand_unsigned_long = np.random.randint(0, 2 ** passwd_length, size=(batch_size,), dtype=np.uint64)
        repeated = np.any(unsigned_long - rand_unsigned_long == 0)
    rand_binary, rand_dis_target = unsigned_long_to_binary_repr(rand_unsigned_long, passwd_length)
    rand_z = torch.from_numpy(rand_binary).to(device)
    rand_dis_target = torch.from_numpy(rand_dis_target).to(device)

    if not inv:
        if use_minus_one is True:
            z = (z - 0.5) * 2
            rand_z = (rand_z - 0.5) * 2
        elif use_minus_one == 'half':
            z -= 0.5
            rand_z -= 0.5
        elif use_minus_one == 'one_fourth':
            z = (z - 0.5) / 2
            rand_z = (z - 0.5) / 2
        return z, dis_target, rand_z, rand_dis_target
    else:
        inv_unsigned_long = 2 ** passwd_length - 1 - unsigned_long
        inv_binary, inv_dis_target = unsigned_long_to_binary_repr(inv_unsigned_long, passwd_length)

        inv_z = torch.from_numpy(inv_binary).to(device)
        inv_dis_target = torch.from_numpy(inv_dis_target).to(device)

        repeated = True
        while repeated:
            rand_inv_unsigned_long = np.random.randint(0, 2 ** passwd_length, size=(batch_size,), dtype=np.uint64)
            repeated = np.any(inv_unsigned_long - rand_inv_unsigned_long == 0)
        rand_inv_binary, rand_inv_dis_target = unsigned_long_to_binary_repr(rand_inv_unsigned_long, passwd_length)
        rand_inv_z = torch.from_numpy(rand_inv_binary).to(device)
        rand_inv_dis_target = torch.from_numpy(rand_inv_dis_target).to(device)

        if gen_random_WR:
            repeated = True
            while repeated:
                rand_inv_2nd_unsigned_long = np.random.randint(0, 2 ** passwd_length, size=(batch_size,),
                                                               dtype=np.uint64)
                repeated = np.any(inv_unsigned_long - rand_inv_2nd_unsigned_long == 0) or np.any(
                    rand_inv_unsigned_long - rand_inv_2nd_unsigned_long == 0)
            rand_inv_2nd_binary, rand_inv_2nd_dis_target = unsigned_long_to_binary_repr(
                rand_inv_2nd_unsigned_long,
                passwd_length)
            rand_inv_2nd_z = torch.from_numpy(rand_inv_2nd_binary).to(device)
            rand_inv_2nd_dis_target = torch.from_numpy(rand_inv_2nd_dis_target).to(device)

        if use_minus_one is True:
            z = (z - 0.5) * 2
            rand_z = (rand_z - 0.5) * 2
            inv_z = z * -1.
            rand_inv_z = (rand_inv_z - 0.5) * 2
            if gen_random_WR:
                rand_inv_2nd_z = (rand_inv_2nd_z - 0.5) * 2
        elif use_minus_one == 'no':
            pass
        elif use_minus_one == 'half':
            z -= 0.5
            rand_z -= 0.5
            inv_z -= 0.5
            rand_inv_z -= 0.5
            if gen_random_WR:
                rand_inv_2nd_z -= 0.5

        elif use_minus_one == 'one_fourth':
            z = (z - 0.5) / 2 # 0 -> -0.25, 1-> 0.25
            rand_z = (rand_z - 0.5) / 2
            inv_z = (inv_z - 0.5) / 2
            rand_inv_z = (rand_inv_z - 0.5) / 2
            if gen_random_WR:
                rand_inv_2nd_z = (rand_inv_2nd_z - 0.5) / 2

        if gen_random_WR:
            return z, dis_target, rand_z, rand_dis_target, \
                   inv_z, inv_dis_target, rand_inv_z, rand_inv_dis_target, rand_inv_2nd_z, rand_inv_2nd_dis_target
        return z, dis_target, rand_z, rand_dis_target, \
               inv_z, inv_dis_target, rand_inv_z, rand_inv_dis_target, None, None

