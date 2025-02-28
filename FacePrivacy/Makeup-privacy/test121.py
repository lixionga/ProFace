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
test_models = {}
for model_name in opt.test_models_list:
    if model_name == 'ir152':
        test_models[model_name] = []
        test_models[model_name].append((112, 112))
        fr_model = ir152.IR_152((112, 112))
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/ir152.pth'))
        fr_model.to(device)
        fr_model.eval()
        test_models[model_name].append(fr_model)
    if model_name == 'irse50':
        test_models[model_name] = []
        test_models[model_name].append((112, 112))
        fr_model = irse.Backbone(50, 0.6, 'ir_se')
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/irse50.pth'))
        fr_model.to(device)
        fr_model.eval()
        test_models[model_name].append(fr_model)
    if model_name == 'facenet':
        test_models[model_name] = []
        test_models[model_name].append((160, 160))
        fr_model = facenet.InceptionResnetV1(num_classes=8631, device=device)
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/facenet.pth'))
        fr_model.to(device)
        fr_model.eval()
        test_models[model_name].append(fr_model)
    if model_name == 'mobile_face':
        test_models[model_name] = []
        test_models[model_name].append((112, 112))
        fr_model = irse.MobileFaceNet(512)
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/mobile_face.pth'))
        fr_model.to(device)
        fr_model.eval()
        test_models[model_name].append(fr_model)

th_dict = {'ir152': (0.094632, 0.166788, 0.227922), 'irse50': (0.144840, 0.241045, 0.312703),
            'facenet': (0.256587, 0.409131, 0.591191), 'mobile_face': (0.183635, 0.301611, 0.380878)}

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])

trans = transforms.Compose([transforms.ToTensor(),normalize])

def denorm(tensor):
    device = tensor.device
    std = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

def norm(tensor):
    device = tensor.device
    std = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1).to(device)
    res = torch.clamp((tensor-mean) / std, 0, 1)
    return res

def cos_similarity( emb_before_pasted, emb_target_img):
    """
    :param emb_before_pasted: feature embedding for the generated adv-makeup face images
    :param emb_target_img: feature embedding for the victim target image
    :return: cosine similarity between two face embeddings
    """
    return torch.mean(torch.sum(torch.mul(emb_target_img, emb_before_pasted), dim=1)
                      / emb_target_img.norm(dim=1) / emb_before_pasted.norm(dim=1))
