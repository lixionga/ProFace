# system libraries
import os, sys
import os.path as osp
import time
import numpy as np
import torchvision
from PIL import Image
import gc
from collections import OrderedDict

import torch
import torchvision.transforms as transforms
# from utils.visualizer import Visualizer
# libraries within this package
from torchvision.transforms import InterpolationMode

import config
from network.noise_layers.face_identity_transformer_master.cmd_args import parse_args
from network.noise_layers.face_identity_transformer_master.utils.util import generate_code
from network.noise_layers.face_identity_transformer_master.utils.visualizer import Visualizer

# sys.path.append(os.path.join('/home/ysc/HiNet/', 'face_identity_transformer_master'))
from network.noise_layers.face_identity_transformer_master import models


class FITSwap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.args = parse_args("/home/ysc/HiNet/face_identity_transformer_master/main.yaml")
        self.args.during_training = False

        self.args.gpu_ids = [0]
        # self.args.device = torch.device('cuda:0')
        self.args.device = torch.device(config.device)

        # 初始化换脸的target datasets
        self.resize_down = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

        self.resize_up = transforms.Compose([
            transforms.Resize((256, 256)),
        ])

        # self.args.test_size = self.args.batch_size // 4 * len(self.args.gpu_ids)
        # test_size = 6

        # add timestamp to ckpt_dir
        self.args.timestamp = time.strftime('%m%d%H%M%S', time.localtime())
        self.args.ckpt_dir += '_' + self.args.timestamp

        # -------------------- create model --------------------
        self.model_dict = {}

        G_input_nc = self.args.input_nc + self.args.passwd_length
        self.model_dict['G'] = models.define_G(G_input_nc, self.args.output_nc,
                                               self.args.ngf, self.args.which_model_netG, self.args.n_downsample_G,
                                               self.args.normG, self.args.dropout,
                                               self.args.init_type, self.args.init_gain,
                                               self.args.passwd_length,
                                               use_leaky=self.args.use_leakyG,
                                               use_resize_conv=self.args.use_resize_conv,
                                               padding_type=self.args.padding_type)
        self.model_dict['G_nets'] = [self.model_dict['G']]

        resume = '/home/ysc/HiNet/face_identity_transformer_master/checkpoints/official/checkpoint_13_iter8193.pth.tar'

        # -------------------- resume --------------------
        if resume:
            if osp.isfile(resume):
                checkpoint = torch.load(resume, map_location='cpu')
                name = 'G'
                net = self.model_dict[name]
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                net.load_state_dict(checkpoint['state_dict_' + name])
                print("=> loaded checkpoints '{}' (epoch {})".format(resume, checkpoint['epoch']))
            else:
                print("=> no checkpoints found at '{}'".format(resume))
            gc.collect()
            torch.cuda.empty_cache()

    def forward(self, x):
        self.model_dict['G'].train()

        x = self.resize_down(x)
        x = x.repeat(32, 1, 1, 1)

        with torch.no_grad():
            z, dis_target, \
            rand_z, rand_dis_target, \
            inv_z, inv_dis_target, \
            rand_inv_z, rand_inv_dis_target, _, _ = generate_code(16,
                                                                  x.size()[0],  # batch_size 输入的batch尽量越大越好
                                                                  "cuda:0",
                                                                  inv=True,
                                                                  use_minus_one="half",
                                                                  gen_random_WR=False)
            fake = self.model_dict['G'](x, z)

        fake = self.resize_up(fake)

        fake = fake[0].unsqueeze(0)

        return fake
