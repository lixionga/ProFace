import os
import time
from PIL import Image
import sys

import torch.nn.functional as F
import torch
import torchvision.transforms as transforms

from torchvision.utils import save_image

from tqdm import tqdm
from util.util import *
from options.base_options import BaseOptions
from models.pix2pix_model import Pix2PixModel
from models.networks.sync_batchnorm import DataParallelWithCallback
from models.networks.face_parsing.parsing_model import BiSeNet
from options.demo_options import DemoOptions

import Pretrained_FR_Models.irse as irse
import Pretrained_FR_Models.facenet as facenet
import Pretrained_FR_Models.ir152 as ir152


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--test_name', type=str, default='1_facenet_multiscale=2', help='Overridden description for test',dest='name')
        parser.add_argument("--source_dir", default="./Dataset-test/CelebA-HQ",help="path to source images")
        parser.add_argument("--reference_dir", default="./Dataset-test/reference",help="path to reference images")
        parser.add_argument("--save_path", default="./LADN_1_irse50_multiscale=2",help="path to generated images")
        parser.add_argument('--which_epoch', type=str, default='latest',help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--beyond_mt', default='True',help='Want to transfer images that are not included in MT dataset, make sure this is Ture')
        parser.add_argument('--demo_mode', type=str, default='normal',help='normal|interpolate|removal|multiple_refs|partly')
        parser.add_argument("--target_image", default="./Dataset-test/test_target_image/test/047073.jpg", help="path to target images")
        parser.add_argument("--test_models_list",type=list, default=['irse50', 'facenet', 'mobile_face', 'ir152'], help="fr models")
        parser.add_argument("--log_path",type=str, default="./log", help="fr models")
        #image quality degradation
        parser.add_argument("--gaussian_blur", default=False, help="image quality degradation")
        parser.add_argument("--gaussian_noise", default=False, help="image quality degradation")
        parser.add_argument("--image_jpeg", default=False, help="image quality degradation")

        parser.add_argument("--kernel_sigma_list",type=list, default=[((5, 5), 1)], help="image quality degradation")#[((5, 5), 1), ((7, 7), 2), ((9, 9), 3), ((11, 11), 4)]
        parser.add_argument("--mean_std_list",type=list, default=[((0, 0.05))], help="image quality degradation")#[((0, 0.05)), ((0, 0.10)), ((0, 0.15)), ((0, 0.20))]
        parser.add_argument("--quality_list",type=list, default=[80], help="image quality degradation")#[80,85,90,95]
        self.isTrain = False
        return parser
opt = TestOptions().parse()

device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids[0] >= 0 else torch.device('cpu')
