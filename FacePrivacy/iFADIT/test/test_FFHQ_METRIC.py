import os
os.environ['CUDA_VISIBLE_DEVICES']='5'
import sys
# sys.path.append('/home/yl/lk/code/RiDDLE-master/')
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
# from utils.image_processing import input_trans, normalize
# import config.config_test as c
from tqdm import tqdm
from torchmetrics import StructuralSimilarityIndexMeasure, MeanSquaredError, PeakSignalNoiseRatio,MeanAbsoluteError

from loss_functions import lpips_loss
import torchvision.transforms as transforms
import torch.nn.functional as F
from calc_IOU import calculate_iou


def normalize(x: torch.Tensor, adaptive=False):
    _min, _max = -1, 1
    if adaptive:
        _min, _max = x.min(), x.max()
    x_norm = (x - _min) / (_max - _min)
    return x_norm

input_trans = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

DIR_HOME = os.path.expanduser("~")
DIR_THIS_PROJECT = os.path.dirname(os.path.realpath(__file__))
DIR_PROJECT ='/home/yl/lk/code/RiDDLE-master'

print("Hello")
device = 'cuda:0'
    # 'CelebA': os.path.join('/home/yl/lk/code/RiDDLE-master/test_data/CelebA'),
    # 'LFW': os.path.join('/home/yl/lk/code/RiDDLE-master/test_data/LFW'),
    # 'VGGFace2': os.path.join('/home/yl/lk/code/RiDDLE-master/test_data/VGGFace2'),


test_datasets = {
    'CelebA': os.path.join('/home/yl/lk/code/RiDDLE-master/test_data/CelebA'),
    # 'LFW': os.path.join('/home/yl/lk/code/RiDDLE-master/test_data/LFW'),
    # 'VGGFace2': os.path.join('/home/yl/lk/code/RiDDLE-master/test_data/VGGFace2'),
    # 'FFHQ': os.path.join('/home/yl/lk/code/RiDDLE-master/test_data/FFHQ')
}
batch_size = 1
epochs = 50
start_epoch, epoch_iter = 1, 0
workers = 16
max_batch = np.inf
embedder_model_path = None

STYLEGAN_PATH = "/home/yl/lk/code/ID-dise4/stylegan2-ffhq-256.pt"


E4E_PATH = "/home/yl/lk/code/RiDDLE-master/e4e_ffhq_encode_256.pt"
MAPPER_PATH = "/home/yl/lk/code/RiDDLE-master/pretrained_models/iteration_90000.pt"
torch.backends.cudnn.enabled = False

########################
# Prepare DE-ID model

import sys
sys.path.append('/home/yl/lk/code/ID-dise4')
sys.path.append('/home/yl/lk/code/ID-dise4/Models')
BASE_PATH = '/home/yl/lk/code/ID-dise4/'
MOBILE_FACE_NET_WEIGHTS_PATH = BASE_PATH + 'mobilefacenet_model_best.pth.tar'

# GENERATOR_WEIGHTS_PATH = BASE_PATH + '550000.pt'
GENERATOR_WEIGHTS_PATH = "/home/yl/lk/code/ID-dise4/stylegan2-ffhq-256.pt"

print(f'GENERATOR_WEIGHTS_PATH = {GENERATOR_WEIGHTS_PATH}')

# GENERATOR_WEIGHTS_PATH = '/home/yl/lk/code/ID-dise4/celeba_hq_8x8_20M_revised.pt'# 用不了
E_ID_LOSS_PATH = BASE_PATH + 'model_ir_se50.pth'

# #LK_DATA FFHQ
# IMAGE_DATA_DIR ='/home/yl/lk/code/ID-dise4/fake/small_image/0/'
# W_DATA_DIR = '/home/yl/lk/code/ID-dise4/fake/small_w/0/'
# MASK_DIR = '/home/yl/lk/code/ID-dise4/fake/small_mask/0/'
# TEST_FILE_NAME = 'IDSF-FFHQ'
# DATA_NAME = 'FFHQ*'

# IMAGE_DATA_DIR ='/home/yl/lk/code/ID-dise4/test_data/FFHQ/images/'
# MASK_DIR = '/home/yl/lk/code/ID-dise4/test_data/FFHQ/masks/'
# TEST_FILE_NAME = 'IDSF-FFHQ'
# DATA_NAME = 'FFHQ'



# TEST_DATA LFW 
# IMAGE_DATA_DIR ='/home/yl/lk/code/ID-dise4/test_data/LFW/images/'
# MASK_DIR = '/home/yl/lk/code/ID-dise4/test_data/LFW/masks/'
# TEST_FILE_NAME = 'IDSF-LFW'
# DATA_NAME = 'LFW'

#TEST_DATA CLELEBA_cropped
# IMAGE_DATA_DIR ='/home/yl/lk/code/ID-dise4/test_data/CelebA/images_crop_224/0/'
# MASK_DIR = '/home/yl/lk/code/ID-dise4/test_data/CelebA/masks_crop_224/0/'
# TEST_FILE_NAME = 'IDSF-CelebA'
# DATA_NAME = 'crop_CelebA'

# TEST_DATA CLELEBA
# IMAGE_DATA_DIR ='/home/yl/lk/code/ID-dise4/test_data/CelebA/images/0/'
# MASK_DIR = '/home/yl/lk/code/ID-dise4/test_data/CelebA/masks/0/'
# TEST_FILE_NAME = 'IDSF-CelebA'
# DATA_NAME = 'crop_CelebA'

# # TEST_DIVERSITY
# IMAGE_DATA_DIR ='/home/yl/lk/code/RiDDLE-master/test_data/Diversity/data/'
# MASK_DIR = '/home/yl/lk/code/RiDDLE-master/test_data/Diversity/mask/'
# TEST_FILE_NAME = 'IDSF-CelebA'
# DATA_NAME = 'Diversity_CelebA'

#TEST_ANONYMIZED
# IMAGE_DATA_DIR ='/home/yl/lk/code/ID-disen_shows/anonymized_shows/TEST_DATA_ver_PNG/data/'
# MASK_DIR = '/home/yl/lk/code/ID-disen_shows/anonymized_shows/TEST_DATA_ver_PNG/mask/'
# TEST_FILE_NAME = 'IDSF-FFHQ'
# DATA_NAME = 'Anonymized_FFHQ'


#LK_ TEST T-SNE ID

# IMAGE_DATA_DIR ='/home/yl/lk/code/ID-dise4/test_data/SNE/small_image/0/'
# MASK_DIR = '/home/yl/lk/code/ID-dise4/test_data/SNE/small_mask/0/'
# TEST_FILE_NAME = 'IDSF-SNE'
# DATA_NAME = 'T-SNE'
# use_SNE =   True

# #CIAGAN T-SNE TEST
# IMAGE_DATA_DIR ='/home/yl/lk/code/ID-dise4/test_data/SNE_CIAGAN/fake/small_image/0/'
# MASK_DIR = '/home/yl/lk/code/ID-dise4/test_data/SNE_CIAGAN/fake/small_mask/0/'
# TEST_FILE_NAME = 'CIAGAN-SNE'
# DATA_NAME = 'CIAGAN-T-SNE'
# use_SNE =   True

#TEST_FFHQ* 可用于复原和解耦测试
W_DATA_DIR = '/userHOME/yl/lk_data/face_disc/small_w/'
IMAGE_DATA_DIR ='/userHOME/yl/lk_data/face_disc/small_image/'
MASK_DIR = '/userHOME/yl/lk_data/face_disc/small_mask/'
TEST_FILE_NAME = 'IDSF-FFHQ*'
DATA_NAME = 'Recovery&Disentangled_FFHQ*'



print(f'IMAGE_DATA_DIR = {IMAGE_DATA_DIR}')
print(f'MASK_DIR = {MASK_DIR}')


# MODELS_DIR = BASE_PATH + 'Models/'
MODELS_DIR = '/home/yl/lk/code/ID-dise4/Models/'
from Configs import Global_Config
from Configs.training_config import config, GENERATOR_IMAGE_SIZE
# from Training.trainer import Trainer
from torch.utils.data import DataLoader, random_split
from Models.Encoders.Landmark_Encoder import Landmark_Encoder
from Models.LatentMapper import LatentMapper, FusionMapper,FusionMapper10
from Models.StyleGan2.model import Generator
from Utils.data_utils import ImageWithMaskDataset, cycle_images_to_create_diff_order, get_masked_imgs, ImageAndMask_Dataset
import torch
print(f'torch version is {torch.__version__}, cuda version is {torch.version.cuda}')
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch.utils.data
from Losses import id_loss
import numpy as np
from options.train_options import TrainOptions
from Configs.paths_config import model_paths
from Models.Encoders.psp import pSp
from face_detect import *
import random
import lpips
from torchvision.utils import save_image
from Models.Deghosting.fixed_imgs import get_fixed_imgs
from Models.Deghosting.deghosting import Deghosting

opts = TrainOptions().parse()

def dilate(bin_img, ksize=9):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out


def erode(bin_img, ksize=5):
    out = 1 - dilate(1 - bin_img, ksize)
    return out




#固定随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)
# setup_seed(1111)

input_dims = (512,)
cond_dims = (512,)
def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 256), nn.LeakyReLU(negative_slope=0.01, inplace=False),
                         nn.Linear(256, 128), nn.LeakyReLU(negative_slope=0.01, inplace=False),
                         nn.Linear(128, 128), nn.LeakyReLU(negative_slope=0.01, inplace=False),
                         nn.Linear(128, 256), nn.LeakyReLU(negative_slope=0.01, inplace=False),
                         nn.Linear(256, dims_out))
G_net = Ff.SequenceINN(*input_dims)
for k in range(4):
    G_net.append(Fm.AllInOneBlock, cond=0, cond_shape=cond_dims, subnet_constructor=subnet_fc)
for m in G_net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

# ************************* with passwords ***********************************
def get_concat_vec(id_images, attr_images, id_encoder, attr_encoder, passwords, mode):
    with torch.no_grad():
        if mode == 'forward':
            id_vec = id_encoder.extract_feats((id_images*2.)-1.)
            # print(f'id_vec shape is {id_vec.shape}, passwords shape is {passwords.shape}')
            id_fake, _ = G_net(id_vec, c=[passwords])
        if mode == 'backward':
            id_vec = id_encoder.extract_feats((id_images*2.)-1.)

            # compensated MLP
            id_vec = fuse_mlp(id_vec)
            # print(f'id_vec shape is {id_vec.shape}, passwords shape is {passwords.shape}')
            id_fake, _ = G_net(id_vec, c=[passwords], rev=True)
        attr_vec = attr_encoder(attr_images)
        
        # attr_vec = fuse_attr_mlp(attr_vec)
        # print(f'_________attr_vec {attr_vec.shape}')          
        id_fake_s = id_fake.unsqueeze(1)
        id_fake_s, _ = torch.broadcast_tensors(id_fake_s, attr_vec)
        # print(f'{id_fake_s.shape, attr_vec.shape}')
        test_id_vec = torch.cat((id_fake_s, attr_vec), dim=2)

    return test_id_vec,id_fake_s, attr_vec, torch.broadcast_tensors(id_vec.unsqueeze(1), attr_vec)[0]

def inversion_encoder(id_images, attr_images, id_encoder, attr_encoder):
    with torch.no_grad():
        id_vec = id_encoder.extract_feats((id_images*2.)-1.).to(Global_Config.device)
        attr_vec = attr_encoder(attr_images)       # *torch.Tensor([0.000000001]).to(Global_Config.device)
        id_fake_s = id_vec.unsqueeze(1)
        id_fake_s, _ = torch.broadcast_tensors(id_fake_s, attr_vec)
        test_id_vec = torch.cat((id_fake_s, attr_vec), dim=2)

    return test_id_vec

def de_attr_encoder(id_images, attr_images, id_encoder, attr_encoder):
    with torch.no_grad():
        id_vec = id_encoder.extract_feats((id_images*2.)-1.)
        attr_vec = attr_encoder(attr_images)*torch.Tensor([0]).to(Global_Config.device)
        id_fake_s = id_vec.unsqueeze(1)
        id_fake_s, _ = torch.broadcast_tensors(id_fake_s, attr_vec)
        test_id_vec = torch.cat((id_fake_s, attr_vec), dim=2)

    return test_id_vec
def de_id_encoder(id_images, attr_images, id_encoder, attr_encoder):
    with torch.no_grad():
        id_vec = id_encoder.extract_feats((id_images*2.)-1.)*torch.Tensor([0]).to(Global_Config.device)
        attr_vec = attr_encoder(attr_images)
        id_fake_s = id_vec.unsqueeze(1)
        id_fake_s, _ = torch.broadcast_tensors(id_fake_s, attr_vec)
        test_id_vec = torch.cat((id_fake_s, attr_vec), dim=2)

    return test_id_vec

def loading_pretrianed(pretrained_dict, net):
    net_dict = net.state_dict()
    keys = []
    for k, v in pretrained_dict.items():
        keys.append(k)
    i = 0
    # 自己网络和预训练网络结构一致的层，使用预训练网络对应层的参数初始化
    for k, v in net_dict.items():
        if v.size() == pretrained_dict[keys[i]].size():
            net_dict[k] = pretrained_dict[keys[i]]
            # print(model_dict[k])
            i = i + 1
    net.load_state_dict(net_dict)

id_encoder = id_loss.IDLoss(E_ID_LOSS_PATH)
attr_encoder = pSp(opts)
mlp = LatentMapper()
landmark_encoder = Landmark_Encoder.Encoder_Landmarks(MOBILE_FACE_NET_WEIGHTS_PATH)
generator = Generator(GENERATOR_IMAGE_SIZE, 512, 8)
fuse_mlp = FusionMapper10()
fuse_attr_mlp = FusionMapper10()
loss_fn_vgg = lpips.LPIPS(net='alex')
loss_fn_vgg = loss_fn_vgg.to(Global_Config.device)
loss_mse = torch.nn.MSELoss().to(Global_Config.device).eval()
state_dict = torch.load(GENERATOR_WEIGHTS_PATH)
generator.load_state_dict(state_dict['g_ema'], strict=False)
degh_net = Deghosting(in_size=128, out_size=256, pretrain='/home/yl/lk/code/ID-dise-psp/550000.pt')


from facenet_pytorch import MTCNN, InceptionResnetV1
IMAGE_SIZE = 220
mtcnn = MTCNN(
    image_size=IMAGE_SIZE, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=Global_Config.device
)
to_pil = transforms.ToPILImage(mode='RGB')


adaface_models = {
    'ir_50':"/home/yl/lk/code/AdaFace-master/pretrained/adaface_ir50_ms1mv2.ckpt",
}
sys.path.append('/home/yl/lk/code/AdaFace-master')
import net
from face_alignment import align

def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return tensor

# no cycle
attr_encoder = torch.load('/home/yl/lk/code/ID-dise4/Training/psp/ckptattr_encoder_MKIHUPYWYPCR_2301.pt')
mlp = torch.load('/home/yl/lk/code/ID-dise4/Training/psp/ckptmaper_MKIHUPYWYPCR_23010.pt')






degh_net.load_state_dict(torch.load('/userHOME/yl/lk_data/deghosting_weight/deghosting_weight_1_140.pth'))
parsing_model_path = "/home/yl/lk/code/faceparsing-master/fp_256.pth"


#FULL
G_pretrained_dict = torch.load('/userHOME/yl/lk_data/train_INN10/G_net_ffhq15INN607_20_202528.pth')# train 15 id cos 0.49400532 ssim 0.94676554
fuse_pretrained_dict = torch.load('/userHOME/yl/lk_data/train_INN10/fuse_mlp_ffhq15INN607_20_202528.pth')#zuihao


# G_pretrained_dict = torch.load('/userHOME/yl/lk_data/train_INN10/G_net_ffhq14INN518_7_19560.pth')# 终极版  id cos 0.20419723 ssim 0.93415153
# fuse_pretrained_dict = torch.load('/userHOME/yl/lk_data/train_INN10/fuse_mlp_ffhq14INN518_7_19560.pth')

# attr_encoder = torch.load('/userHOME/yl/lk_data/train_INN10/attr_encoder_ffhq22INN619_2_162702.pt')
# mlp = torch.load('/userHOME/yl/lk_data/train_INN10/mlp_ffhq22INN619_2_162702.pt')
# G_pretrained_dict = torch.load('/userHOME/yl/lk_data/train_INN10/G_net_ffhq22INN619_2_162702.pth')# train 22
# fuse_pretrained_dict = torch.load('/userHOME/yl/lk_data/train_INN10/fuse_mlp_ffhq22INN619_2_162702.pth') #id cos 0.35755122 ssim 0.94144094


# attr_encoder = torch.load('/userHOME/yl/lk_data/train_INN10/attr_encoder_ffhq20INN618_5_191859.pt')# 
# mlp = torch.load('/userHOME/yl/lk_data/train_INN10/mlp_ffhq20INN618_5_191859.pt')
# G_pretrained_dict = torch.load('/userHOME/yl/lk_data/train_INN10/G_net_ffhq20INN618_5_191859.pth')# train 20  cos 0.267479 ssim 0.94127095
# fuse_pretrained_dict = torch.load('/userHOME/yl/lk_data/train_INN10/fuse_mlp_ffhq20INN618_5_191859.pth')

# G_pretrained_dict = torch.load('/userHOME/yl/lk_data/train_INN10/G_net_ffhq16INN530_13_37854.pth')# train 16 cos 0.10054752 ssim 0.90175724
# fuse_pretrained_dict = torch.load('/userHOME/yl/lk_data/train_INN10/fuse_mlp_ffhq16INN530_13_37854.pth')
 
# G_pretrained_dict = torch.load('/userHOME/yl/lk_data/train_INN10/G_net_ffhq16INN603_27_36669.pth')# train 16 id cos 0.09148127 ssim 0.90100765
# fuse_pretrained_dict = torch.load('/userHOME/yl/lk_data/train_INN10/fuse_mlp_ffhq16INN603_27_36669.pth')

# G_pretrained_dict = torch.load('/userHOME/yl/lk_data/train_INN10/G_net_ffhq18INN614_15_124876.pth')# train 18 不能正常运行 是INN的问题？
# fuse_pretrained_dict = torch.load('/userHOME/yl/lk_data/train_INN10/fuse_mlp_ffhq18INN614_15_124876.pth')

# G_pretrained_dict = torch.load('/userHOME/yl/lk_data/train_INN10/G_net_ffhq19INN614_5_133062.pth')# train 19 不能正常运行 是INN的问题？MLP?
# fuse_pretrained_dict = torch.load('/userHOME/yl/lk_data/train_INN10/fuse_mlp_ffhq19INN614_5_133062.pth')

# G_pretrained_dict = torch.load('/userHOME/yl/lk_data/train_INN10/G_net_ffhq20INN615_13_20764.pth')# train 20 未使用正则化 不能正常运行 是INN的问题？MLP?
# fuse_pretrained_dict = torch.load('/userHOME/yl/lk_data/train_INN10/fuse_mlp_ffhq20INN615_13_20764.pth')

Alation_Path = 'FULL'
print(f'loading model is full')

# #w/o privacy loss
# G_pretrained_dict = torch.load('/userHOME/yl/lk_data/ablation_weights/wo_pri/G_net_3_199854.pth')# train 15
# fuse_pretrained_dict = torch.load('/userHOME/yl/lk_data/ablation_weights/wo_pri/fuse_mlp_3_199854.pth')#zuihao

#w/o parse loss
# G_pretrained_dict = torch.load('/userHOME/yl/lk_data/ablation_weights/wo_parse/G_net_3_190584.pth')# train 15
# fuse_pretrained_dict = torch.load('/userHOME/yl/lk_data/ablation_weights/wo_parse/fuse_mlp_3_190584.pth')#zuihao


# #w/o div
# G_pretrained_dict = torch.load('/userHOME/yl/lk_data/ablation_weights/wo_DIV/G_net_18_187499.pth')
# fuse_pretrained_dict = torch.load('/userHOME/yl/lk_data/ablation_weights/wo_DIV/fuse_mlp_18_187499.pth')
# Alation_Path = 'DIV_WO'
# print(f'loading model is w/o div')

#w/o img
# G_pretrained_dict = torch.load('/userHOME/yl/lk_data/ablation_weights/wo_IMG/G_net_17_209988.pth')
# fuse_pretrained_dict = torch.load('/userHOME/yl/lk_data/ablation_weights/wo_IMG/fuse_mlp_17_209988.pth')
# Alation_Path = 'IMG_WO'
# print(f'loading model is w/o img')

#w/o mlp
# G_pretrained_dict = torch.load('/userHOME/yl/lk_data/train_INN10/G_net_ffhq15INN607_20_202528.pth')# train 15
# Alation_Path = 'MLP_WO'
# print(f'loading model is w/o MLP')


# w/o one_stage
# attr_encoder_dict = torch.load('/userHOME/yl/lk_data/ablation_weights/wo_one_stage/attr_encoder_19_405036.pth')
# mlp_dict = torch.load('/userHOME/yl/lk_data/ablation_weights/wo_one_stage/mlp_19_405036.pth')
# G_pretrained_dict = torch.load('/userHOME/yl/lk_data/ablation_weights/wo_one_stage/G_net_19_405036.pth')
# fuse_pretrained_dict = torch.load('/userHOME/yl/lk_data/ablation_weights/wo_one_stage/fuse_mlp_19_405036.pth')
# Alation_Path = 'ONE_STAGE_WO'
# # loading_pretrianed(attr_encoder_dict, attr_encoder)
# # loading_pretrianed(mlp_dict, mlp)
# print(f'loading model is one_stage')


loading_pretrianed(G_pretrained_dict, G_net)
if Alation_Path != 'MLP_WO':
    loading_pretrianed(fuse_pretrained_dict, fuse_mlp)

# attr_dict = attr_encoder.state_dict()
# keys = []
# for k, v in attr_encoder_dict.items():
#     keys.append(k)
# i = 0
# # 自己网络和预训练网络结构一致的层，使用预训练网络对应层的参数初始化
# for k, v in attr_dict.items():
#     if v.size() == attr_encoder_dict[keys[i]].size():
#         attr_dict[k] = attr_encoder_dict[keys[i]]
#         # print(model_dict[k])
#         i = i + 1
# attr_encoder.load_state_dict(attr_dict)

# mlp_dict = mlp.state_dict()
# keys = []
# for k, v in G_pretrained_dict.items():
#     keys.append(k)
# i = 0
# # 自己网络和预训练网络结构一致的层，使用预训练网络对应层的参数初始化
# for k, v in mlp_dict.items():
#     if v.size() == G_pretrained_dict[keys[i]].size():
#         mlp_dict[k] = G_pretrained_dict[keys[i]]
#         # print(model_dict[k])
#         i = i + 1
# mlp.load_state_dict(mlp_dict)

# G_dict = G_net.state_dict()
# keys = []
# for k, v in G_pretrained_dict.items():
#     keys.append(k)
# i = 0
# # 自己网络和预训练网络结构一致的层，使用预训练网络对应层的参数初始化
# for k, v in G_dict.items():
#     if v.size() == G_pretrained_dict[keys[i]].size():
#         G_dict[k] = G_pretrained_dict[keys[i]]
#         # print(model_dict[k])
#         i = i + 1
# G_net.load_state_dict(G_dict)

# #id
# fuse_dict = fuse_mlp.state_dict()
# keys = []
# for k, v in fuse_pretrained_dict.items():
#     keys.append(k)
# i = 0
# # 自己网络和预训练网络结构一致的层，使用预训练网络对应层的参数初始化
# for k, v in fuse_dict.items():
#     if v.size() == fuse_pretrained_dict[keys[i]].size():
#         fuse_dict[k] = fuse_pretrained_dict[keys[i]]
#         # print(model_dict[k])
#         i = i + 1
# fuse_mlp.load_state_dict(fuse_dict)


id_encoder = id_encoder.to(Global_Config.device)
attr_encoder = attr_encoder.to(Global_Config.device)
mlp = mlp.to(Global_Config.device)
generator = generator.to(Global_Config.device)
landmark_encoder = landmark_encoder.to(Global_Config.device)
G_net = G_net.to(Global_Config.device)
fuse_mlp = fuse_mlp.to(Global_Config.device)
degh_net = degh_net.to(Global_Config.device)
fuse_attr_mlp = fuse_attr_mlp.to(Global_Config.device)



id_encoder = id_encoder.eval()
attr_encoder = attr_encoder.eval()
generator = generator.eval()
mlp = mlp.eval()
landmark_encoder = landmark_encoder.eval()
G_net = G_net.eval()
fuse_mlp = fuse_mlp.eval()
degh_net = degh_net.eval()
fuse_attr_mlp = fuse_attr_mlp.eval()




#SaveImage
def SaveImage(input_imgs, save_path):
    """
    input_imgs :except tensor [B,C,H,W]
    save_path: path to saving imgs
    """
    batch,_,_,_ = input_imgs.shape

    for idx in range(batch):
        save_image(input_imgs[idx], save_path+f'{idx}.png')

#load data
# w_image_dataset = ImageWithMaskDataset(W_DATA_DIR, IMAGE_DATA_DIR, MASK_DIR)
# train_size = int(config['train_precentege'] * len(w_image_dataset))
# test_size = len(w_image_dataset) - train_size
# train_data, test_data = random_split(w_image_dataset, [train_size, test_size])

# batch_data_loader = DataLoader(dataset=w_image_dataset, batch_size=config['batchSize'], shuffle=False,drop_last=True)

############### 用于测试复原 ###############

if DATA_NAME == 'Recovery&Disentangled_FFHQ*':
    w_image_dataset = ImageWithMaskDataset(W_DATA_DIR, IMAGE_DATA_DIR, MASK_DIR)
    train_size = int(config['train_precentege'] * len(w_image_dataset))
    test_size = len(w_image_dataset) - train_size
    train_data, test_data = random_split(w_image_dataset, [train_size, test_size])
    batch_data_loader = DataLoader(dataset=test_data, batch_size=config['batchSize'], shuffle=False,drop_last=True)
else:
    w_image_dataset = ImageAndMask_Dataset(IMAGE_DATA_DIR, MASK_DIR)
    batch_data_loader = DataLoader(dataset=w_image_dataset, batch_size=config['batchSize'], shuffle=False,drop_last=False)


########################
# Prepare metrics
lpips_loss.to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
mse = MeanSquaredError().to(device)
psnr = PeakSignalNoiseRatio().to(device)
mae = MeanAbsoluteError().to(device)

#######################
# Prepare image dataset

rec_ssim_list = []
rec_psnr_list = []
rec_lpips_list = []
rec_mse_list = []
Face_detection_list = []
Bounding_box_distance_list = []
Landmark_distance_list = []
arc_cos_similarity_list = []
vgg_cos_similarity_list = []
casia_cos_similarity_list = []
anon_mse_list = []
anon_mae_list = []
arc_anon_mse_list = []
arc_anon_mae_list = []
vgg_anon_mse_list = []
vgg_anon_mae_list =[]
fake_ssim_list = []
fake_mse_list = []
fake_psnr_list = []
fake_lpips_list =[]
Zfake_cos_similarity_list =[]
fake2_cos_similarity_list =[]
Zrecon_cos_similarity_list =[]
fakeZ_cos_similarity_list =[]
ada_cos_similarity_list = []
ada_anon_mae_list = []
recon_arc_cos_similarity_list = []
for i_batch, data_batch in tqdm(enumerate(batch_data_loader)):
    if DATA_NAME == 'Recovery&Disentangled_FFHQ*':
        _, images, masks = data_batch
    else:
        images, masks = data_batch
    if i_batch > 19 :
        break

    test_id_images = images.to(Global_Config.device)
    x_ori = test_id_images
    test_masks = masks.to(Global_Config.device)

    use_512bitpasswords = True

    # set up passwords


    if use_512bitpasswords == True:

        passwords_fir = (torch.rand((1,512),device=Global_Config.device)+1.)/2.
        passwords_fir, _ = torch.broadcast_tensors(passwords_fir, torch.rand((config['batchSize'],512),device=Global_Config.device))
        passwords_fir = generator.style(passwords_fir)

        passwords_thr = (torch.rand((1,512),device=Global_Config.device)+1.)/2.   
        passwords_thr, _ = torch.broadcast_tensors(passwords_thr, torch.rand((config['batchSize'],512),device=Global_Config.device))
        passwords_thr = generator.style(passwords_thr)

        while 1:
            passwords_sec = (torch.rand((1,512),device=Global_Config.device)+1.)/2.   
            passwords_sec, _ = torch.broadcast_tensors(passwords_sec, torch.rand((config['batchSize'],512),device=Global_Config.device))
            passwords_sec = generator.style(passwords_sec)
            if passwords_fir[0][0] != passwords_sec[0][0] or passwords_fir[0][1] != passwords_sec[0][1]:
                break            

    else:

        ValueError()
    # config
    with_passwords = 1
    auto = 1
    ############distangled############



    if with_passwords == True:
        
# #########################test  sne########################
#         if use_SNE == True:
#             with torch.no_grad():
#                 id_vec = id_encoder.extract_feats((test_id_images*2.)-1.)
#                 for sne_idx in range(100):

#                     ps_sne = (torch.rand((1,512),device=Global_Config.device)+1.)/2.
#                     ps_sne, _ = torch.broadcast_tensors(ps_sne, torch.rand((config['batchSize'],512),device=Global_Config.device))
#                     ps_sne = generator.style(ps_sne)
                    
#                     id_fake, _ = G_net(id_vec, c=[ps_sne])
#                     if sne_idx == 0:
#                         ids = id_fake
#                     else :
#                     #     print(f'id_fake shape = {id_fake.shape},ids shape = {ids.shape} ')
#                     #     a
#                         ids = torch.cat((ids,id_fake),dim=0)
#                 ids = ids.cpu().numpy()
#                 print(f'id shape = {ids.shape}')
#                 np.save(f'id_{i_batch}', ids)

# #########################test  sne########################


# #########################kffa########################
        # with torch.no_grad():
        #     # id_vec = id_encoder.extract_feats((test_id_images*2.)-1.)
        #     # cos_list = []
        #     for sne_idx in range(50):
        #         ps_sne = None
        #         ps_sne = (torch.rand((1,512),device=Global_Config.device)+1.)/2.
        #         ps_sne, _ = torch.broadcast_tensors(ps_sne, torch.rand((config['batchSize'],512),device=Global_Config.device))
        #         ps_sne = generator.style(ps_sne)
            
        #         kffa_w, _, _, _ = get_concat_vec(test_id_images, test_id_images, id_encoder, attr_encoder, ps_sne, mode='forward')
        #         # kffa_id = torch.mean(kffa_id, dim=1)
        #         with torch.no_grad():
        #             kffa_code = mlp(kffa_w)
        #             kffa_imgs_tensor, _ = generator([kffa_code], input_is_latent=True, return_latents=False, randomize_noise=False)

        #         kffa_imgs = get_masked_imgs((kffa_imgs_tensor+1)/2, test_id_images, test_masks)

        
        #         save_image(kffa_imgs, os.path.join(f'/home/yl/lk/code/ID-disen_shows/ablation_shows/{Alation_Path}/KFFA/imgs/{str(i_batch+1)}_Ours_{str(sne_idx+1)}.jpg'))
        #         # save_image(kffa_imgs, os.path.join(f'/home/yl/lk/code/ID-disen_shows/anonymized_shows/Ours/{str(i_batch+1)}_Ours_{str(sne_idx+1)}.jpg'))


            #     concat_rand_recon_vec_cycled, rand_recon_id, rand_recon_attr, rand_recon_id_before = get_concat_vec(kffa_imgs_tensor,kffa_imgs_tensor,id_encoder, attr_encoder,passwords_sec,mode='backward')
            #     with torch.no_grad():
            #         mapped_concat_rand_recon_vec_cycled = mlp(concat_rand_recon_vec_cycled)   
            #         other_recon_imgs, _ = generator([mapped_concat_rand_recon_vec_cycled], input_is_latent=True, return_latents=True, randomize_noise=False)
            #         rand_recon = get_masked_imgs((other_recon_imgs+1)/2, test_id_images, test_masks)
            #         save_image(kffa_imgs, os.path.join(f'/home/yl/lk/code/ID-disen_shows/anonymized_shows/Ours/{str(i_batch+1)}_Ours_wrong_recon_{str(sne_idx+1)}.jpg'))
            #     kffa_id = id_encoder.extract_feats(kffa_imgs)
            #     if sne_idx == 0:
            #         ids = kffa_id
            #     else :
            #         ids = torch.cat((ids,kffa_id),dim=0)
            #     # ids = ids.cpu().numpy()
            #     # np.save(f'id_{i_batch}', ids)
            # if i_batch == 0:
            #     total_ids = ids
            # else :
            #     total_ids = torch.cat((total_ids,ids),dim=0)
                
            # print(f'total_ids shape = {total_ids.shape}')

            
                # if sne_idx >0:
                #     cos_similarity = torch.cosine_similarity(ids[sne_idx-1].unsqueeze(0), ids[sne_idx].unsqueeze(0), dim=1)
                #     cos_list.append(cos_similarity)
                #     print(f'cos_similarity = {cos_similarity}')
            
            # print(f'cos_list.mean() = {cos_list.mean()}')
 # #########################kffa########################       

    




        # #inversion
        # concat_inv = inversion_encoder(test_id_images, test_id_images, id_encoder, attr_encoder)
        # with torch.no_grad():
        #     mapped_concat_vec_inv = mlp(concat_inv)
        #     inv_imgs, _ = generator([mapped_concat_vec_inv], input_is_latent=True, return_latents=False, randomize_noise=False)
        #     # inve = (inv_imgs+1.)/2
        #     inve = get_masked_imgs((inv_imgs+1)/2, test_id_images, test_masks)


        #first generated
        concat_vec_cycled_fir, fake_id_fir, orin_attr, fake_id_before = get_concat_vec(test_id_images, test_id_images, id_encoder, attr_encoder, passwords_fir, mode='forward')
        with torch.no_grad():
            mapped_concat_vec_fir = mlp(concat_vec_cycled_fir)
            anonymous_fir_imgs, _ = generator([mapped_concat_vec_fir], input_is_latent=True, return_latents=False, randomize_noise=False)
            generated_images_tensor = get_masked_imgs((anonymous_fir_imgs+1)/2, test_id_images, test_masks)
        
        #random generated
        concat_vec_cycled_sec, fake_id_sec, _, _ = get_concat_vec(test_id_images, test_id_images, id_encoder, attr_encoder, passwords_sec, mode='forward')
        with torch.no_grad():
            mapped_concat_vec_sec = mlp(concat_vec_cycled_sec)
            anonymous_sec_imgs, _ = generator([mapped_concat_vec_sec], input_is_latent=True, return_latents=False, randomize_noise=False)
            generated_sec = get_masked_imgs((anonymous_sec_imgs+1)/2, test_id_images, test_masks)

        # recon
        concat_recon_vec_cycled, recon_id, recon_attr, recon_id_before = get_concat_vec(generated_images_tensor, generated_images_tensor, id_encoder, attr_encoder, passwords_fir, mode='backward')
        with torch.no_grad():
            mapped_concat_recon_vec_cycled = mlp(concat_recon_vec_cycled)
            recon_imgs, _ = generator([mapped_concat_recon_vec_cycled], input_is_latent=True, return_latents=True, randomize_noise=False)
            recon = get_masked_imgs((recon_imgs+1)/2, test_id_images, test_masks)

        
        # # rand recon
        # concat_rand_recon_vec_cycled, rand_recon_id, rand_recon_attr, rand_recon_id_before = get_concat_vec(generated_images_tensor,generated_images_tensor,id_encoder, attr_encoder,passwords_sec,mode='backward')
        # with torch.no_grad():
        #     mapped_concat_rand_recon_vec_cycled = mlp(concat_rand_recon_vec_cycled)   
        #     other_recon_imgs, _ = generator([mapped_concat_rand_recon_vec_cycled], input_is_latent=True, return_latents=True, randomize_noise=False)
        #     rand_recon = get_masked_imgs((other_recon_imgs+1)/2, test_id_images, test_masks)

        # #de_attr
        # concat_de_attr = de_attr_encoder(test_id_images, test_id_images, id_encoder, attr_encoder)
        # with torch.no_grad():
        #     mapped_concat_vec_de_attr = mlp(concat_de_attr)
        #     de_attr, _ = generator([mapped_concat_vec_de_attr], input_is_latent=True, return_latents=False, randomize_noise=False)


        # #de_id
        # concat_de_id = de_id_encoder(test_id_images, test_id_images, id_encoder, attr_encoder)
        # with torch.no_grad():
        #     mapped_concat_vec_de_id = mlp(concat_de_id)
        #     de_id, _ = generator([mapped_concat_vec_de_id], input_is_latent=True, return_latents=False, randomize_noise=False)

      

        save_image(torch.cat((x_ori,generated_images_tensor,generated_sec,recon), dim=0), os.path.join(f'/home/yl/lk/code/ID-dise4/metrics/test_FFHQ_Metric.jpg'))

        # SAVE_IMG_PATH ="/home/yl/lk/code/ID-disen_shows/ablation_shows/ONE_STAGE_WO"





        # #save img 
        # save_image(x_ori, os.path.join(f'{SAVE_IMG_PATH}/TEST_DATA/orig/{str(i_batch+1)}_Orig.jpg'))
        # save_image(inve, os.path.join(f'{SAVE_IMG_PATH}/inversion/{str(i_batch+1)}_Ours_1.jpg'))
        # save_image(generated_images_tensor, os.path.join(f'{SAVE_IMG_PATH}/Anon1/{str(i_batch+1)}_Ours_1.jpg'))
        # save_image(generated_sec, os.path.join(f'{SAVE_IMG_PATH}/Anon2/{str(i_batch+1)}_Ours_1.jpg'))
        # save_image(recon, os.path.join(f'{SAVE_IMG_PATH}/Recon/{str(i_batch+1)}_Ours_1.jpg'))
        # save_image(rand_recon, os.path.join(f'{SAVE_IMG_PATH}/RandRecon/{str(i_batch+1)}_Ours_1.jpg'))
        # save_image(test_masks, os.path.join(f'{SAVE_IMG_PATH}/TEST_DATA/mask/ {str(i_batch+1)}_Orig.jpg'))

        # save_image((de_attr+1.)/2., os.path.join(f'/home/yl/lk/code/ID-disen_shows/Disentangled_shows/Ours/De_attr/{str(i_batch+1)}_Ours_1.jpg'))
        # save_image((de_id+1.)/2., os.path.join(f'/home/yl/lk/code/ID-disen_shows/Disentangled_shows/Ours/De_id/{str(i_batch+1)}_Ours_1.jpg'))
        # save_image(test_masks, os.path.join(f'/home/yl/lk/code/ID-disen_shows/Recovery_shows/TEST_DATA/mask/ {str(i_batch+1)}_Orig.jpg'))
        #save 
        # total_grad = torch.concat((x_ori, generated_images_tensor, recon), dim=0)
        # save_image(total_grad, f'/home/yl/lk/code/ID-dise4/metrics/{str(i_batch)}_total.jpg',nrow=4)
        

        # save_image(x_ori,f'/home/yl/lk/code/ID-dise4/visual/{TEST_FILE_NAME}/{DATA_NAME}_{str(i_batch)}_Orin.png')
        # save_image(generated_images_tensor,f'/home/yl/lk/code/ID-dise4/visual/{TEST_FILE_NAME}/{DATA_NAME}_{str(i_batch)}_Anon.png')
        # save_image((anonymous_fir_imgs+1)/2,f'/home/yl/lk/code/ID-dise4/visual/{TEST_FILE_NAME}/{DATA_NAME}_{str(i_batch)}_nomask_Anon.png')
        # save_image(generated_sec,f'/home/yl/lk/code/ID-dise4/visual/{TEST_FILE_NAME}/{DATA_NAME}_{str(i_batch)}_RandAnon.png')
        # save_image((recon_imgs+1)/2,f'/home/yl/lk/code/ID-dise4/visual/{TEST_FILE_NAME}/{DATA_NAME}_{str(i_batch)}_nomask_Recon.png')
        # save_image(recon,f'/home/yl/lk/code/ID-dise4/visual/{TEST_FILE_NAME}/{DATA_NAME}_{str(i_batch)}_Recon.png')
        # save_image(rand_recon,f'/home/yl/lk/code/ID-dise4/visual/{TEST_FILE_NAME}/{DATA_NAME}_{str(i_batch)}_RandRecon.png')
        # save_image(inve,f'/home/yl/lk/code/ID-dise4/visual/{TEST_FILE_NAME}/{DATA_NAME}_{str(i_batch)}_Inversion.png')

        # save_image(test_masks,f'/home/yl/lk/code/ID-dise4/visual/IDSF-FFHQ/FFHQ_{str(i_batch)}_Mask.png')


# ####################################################### 计算指标
        # Recovery quality x_ori
        rec_ssim_score = ssim(normalize(recon.to(device)), normalize(x_ori.to(device))).cpu().detach().numpy()
        rec_ssim_list.append(rec_ssim_score)

        rec_mse = mse(normalize(recon.to(device)), normalize(x_ori.to(device))).cpu().detach().numpy()
        rec_mse_list.append(rec_mse)

        rec_psnr = psnr(normalize(recon.to(device)), normalize(x_ori.to(device))).cpu().detach().numpy()
        rec_psnr_list.append(rec_psnr)

        rec_lpips = lpips_loss(normalize(recon.to(device)), normalize(x_ori.to(device))).cpu().detach().numpy()
        rec_lpips_list.append(rec_lpips)

        arc_fake_embedding = id_encoder.extract_feats(recon*2-1)# imgs x_ori
        arc_orig_embedding = id_encoder.extract_feats(x_ori*2-1)# fake_save*2-1 fake
        recon_arc_cos_similarity = torch.cosine_similarity(arc_fake_embedding, arc_orig_embedding, dim=1)
        recon_arc_cos_similarity_list.append(recon_arc_cos_similarity.abs().detach().cpu().numpy())  

        # Fake quality fake

        # fake_ssim_score = ssim(normalize(generated_images_tensor.to(device)), normalize(x_ori.to(device))).cpu()
        # fake_ssim_list.append(fake_ssim_score)

        # fake_mse = mse(normalize(generated_images_tensor.to(device)), normalize(x_ori.to(device))).cpu()
        # fake_mse_list.append(fake_mse)

        # fake_psnr = psnr(generated_images_tensor.to(device), x_ori.to(device)).cpu()
        # fake_psnr_list.append(fake_psnr)

        # fake_lpips = lpips_loss(normalize(generated_images_tensor.to(device)), normalize(x_ori.to(device))).detach().cpu()
        # fake_lpips_list.append(fake_lpips)
        

        # from facenet_pytorch import MTCNN, InceptionResnetV1
        # IMAGE_SIZE = 220
        # mtcnn = MTCNN(
        #     image_size=IMAGE_SIZE, margin=0, min_face_size=20,
        #     thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        #     device=Global_Config.device
        # )
        # to_pil = transforms.ToPILImage(mode='RGB')
        #mtcnn ori_bboxes[0][0] 代表检测框左边，ori_bboxes[0][1]代表人脸被检测到的概率， ori_bboxes[0][2]代表人脸关键点
        ori_bboxes = [mtcnn.detect(to_pil(image),landmarks=True) for image in (x_ori+1)/2]
        fake_bboxes = [mtcnn.detect(to_pil(image),landmarks=True) for image in generated_images_tensor] #generated_images_tensor (anonymous_fir_imgs+1)/2
        try:
            Face_detection = (calculate_iou(ori_bboxes[0][0], fake_bboxes[0][0]))
            Face_detection_list.append(Face_detection)
        except Exception as e:
            print(f"An error occurred: {e}")

        Bounding_box_distance = (abs(fake_bboxes[0][1]-ori_bboxes[0][1]))
        Bounding_box_distance_list.append(Bounding_box_distance)

        Landmark_distance = (abs(ori_bboxes[0][2] - fake_bboxes[0][2]))
        Landmark_distance_list.append(Landmark_distance)
        
        #arcface 
        arc_fake_embedding = id_encoder.extract_feats(anonymous_fir_imgs)#generated_images_tensor*2-1 anonymous_fir_imgs
        arc_orig_embedding = id_encoder.extract_feats(x_ori*2-1)
        arc_cos_similarity = torch.cosine_similarity(arc_fake_embedding, arc_orig_embedding, dim=1) #(fake_id_fir.mean(1), fake_id_before.mean(1), dim=1)
        arc_cos_similarity_list.append(arc_cos_similarity.detach().cpu().numpy())
            
        # ada_model = load_pretrained_model('ir_50').to(Global_Config.device)
        # def fix_align(input_imgs):
        #     input_imgs = input_imgs[:, :, 35:223, 32:220]
        #     with torch.no_grad():
        #         face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        #         out_imgs = face_pool(input_imgs)
        #     return out_imgs
        # ada_orig_feature, _ = ada_model(fix_align(generated_images_tensor*2-1)) #generated_images_tensor*2-1 anonymous_fir_imgs
        # ada_fake_feature, _ = ada_model(fix_align(x_ori*2-1))
        # ada_cos_similarity = torch.cosine_similarity(ada_orig_feature, ada_fake_feature, dim=1) #(fake_id_fir.mean(1), fake_id_before.mean(1), dim=1)
        # ada_cos_similarity_list.append(ada_cos_similarity.detach().cpu().numpy())       

        # ada_anon_mae = mae(normalize(ada_orig_feature.to(device)), normalize(ada_fake_feature.to(device))).cpu().detach().numpy()
        # ada_anon_mae_list.append(ada_anon_mae)   


        # # #calc SIT
        # #z,z..
        # Zfake_cos_similarity = torch.cosine_similarity(fake_id_fir.mean(1), fake_id_before.mean(1), dim=1) 
        # Zfake_cos_similarity_list.append(Zfake_cos_similarity.detach().cpu().numpy())

        # #z,zf
        # Zrecon_cos_similarity = torch.cosine_similarity(recon_id.mean(1), fake_id_before.mean(1), dim=1) 
        # Zrecon_cos_similarity_list.append(Zrecon_cos_similarity.detach().cpu().numpy())

        # #z..,zf
        # fake2_cos_similarity = torch.cosine_similarity(recon_id.mean(1), recon_id_before.mean(1), dim=1) 
        # fake2_cos_similarity_list.append(fake2_cos_similarity.detach().cpu().numpy())

        # #z..,z..
        # fakeZ_cos_similarity = torch.cosine_similarity(fake_id_fir.mean(1), recon_id_before.mean(1), dim=1) 
        # fakeZ_cos_similarity_list.append(fakeZ_cos_similarity.detach().cpu().numpy())

        #calc norm feature
        # arc_anon_mse = mse(arc_fake_embedding.to(device), arc_orig_embedding.to(device)).cpu().detach().numpy()
        # arc_anon_mse_list.append(arc_anon_mse)

        # arc_anon_mae = mae(arc_fake_embedding.to(device), arc_orig_embedding.to(device)).cpu().detach().numpy()
        # arc_anon_mae_list.append(arc_anon_mae)     


        # #vggface2
        # vgg_face_net = InceptionResnetV1(pretrained='vggface2').eval().to(Global_Config.device)
        # vgg_fake_cropped = mtcnn(to_pil(generated_images_tensor.reshape(3,256,256)))
        # vgg_fake_embedding = vgg_face_net(vgg_fake_cropped.unsqueeze(0).to(Global_Config.device))
        # vgg_orig_cropped = mtcnn(to_pil(test_id_images.reshape(3,256,256)))
        # vgg_orig_embedding = vgg_face_net(vgg_orig_cropped.unsqueeze(0).to(Global_Config.device))

        # vgg_cost_similarity = torch.cosine_similarity(vgg_fake_embedding, vgg_orig_embedding, dim=1)
        # vgg_cos_similarity_list.append(vgg_cost_similarity.abs().detach().cpu().numpy())


        # # vgg_anon_mse = mse(normalize(vgg_fake_embedding.to(device)), normalize(vgg_orig_embedding.to(device))).cpu().detach().numpy()
        # # vgg_anon_mse_list.append(vgg_anon_mse)
        
        # vgg_anon_mae = mae(normalize(vgg_fake_embedding.to(device)), normalize(vgg_orig_embedding.to(device))).cpu().detach().numpy()
        # vgg_anon_mae_list.append(vgg_anon_mae)                   


#         #casia
#         casia_face_net = InceptionResnetV1(pretrained='casia-webface').eval().to(Global_Config.device)
#         casia_fake_cropped = mtcnn(to_pil((anonymous_fir_imgs.reshape(3,256,256)+1)/2.))
#         casia_fake_embedding = casia_face_net(casia_fake_cropped.unsqueeze(0).to(Global_Config.device))
#         casia_orig_cropped = mtcnn(to_pil(test_id_images.reshape(3,256,256)))
#         casia_orig_embedding = casia_face_net(casia_orig_cropped.unsqueeze(0).to(Global_Config.device))
# #         # print(f'aa')
#         casia_cost_similarity = torch.cosine_similarity(casia_fake_embedding, casia_orig_embedding, dim=1)
#         casia_cos_similarity_list.append(casia_cost_similarity.detach().cpu().numpy())

#         anon_mse = mse(normalize(generated_images_tensor.to(device)), normalize(test_id_images.to(device))).cpu().detach().numpy()
#         anon_mse_list.append(anon_mse)
        
#         anon_mae = mae(normalize(generated_images_tensor.to(device)), normalize(test_id_images.to(device))).cpu().detach().numpy()
#         anon_mae_list.append(anon_mae)


# ##########################################################

        # save_image((imgs + 1) / 2, f'show_imgs/{dname}/batch{str(i_batch)}_orin.jpg')
        # save_image((fake + 1) / 2, f'show_imgs/{dname}/batch{str(i_batch)}_fake.jpg')
        # save_image((recon + 1) / 2, f'show_imgs/{dname}/batch{str(i_batch)}_recon.jpg')
        # save_image((wrong_recon + 1) / 2, f'show_imgs/{dname}/batch{str(i_batch)}_wrong_recon.jpg')


#         # save_image((test_imgs + 1) / 2, f'temp_images/test_{i_batch}.jpg', nrow=4)
#         # save_image((x_ori + 1) / 2, f'temp_images/x_ori_{i_batch}.jpg', nrow=4)
#         # save_image((imgs + 1) / 2, f'temp_images/orig_{i_batch}.jpg', nrow=4)
#         # save_image((fake + 1) / 2, f'temp_images/fake_{i_batch}.jpg', nrow=4)
#         # save_image((rand_fake + 1) / 2, f'temp_images/rand_fake_{i_batch}.jpg', nrow=4)
#         # save_image((recon + 1) / 2, f'temp_images/recon_{i_batch}.jpg', nrow=4)
#         # save_image((wrong_recon + 1) / 2, f'temp_images/wrong_recon_{i_batch}.jpg', nrow=4)

# total_ids = total_ids.cpu().numpy()
# np.save(f'/home/yl/lk/code/ID-dise4/Training/TSNE_OUT/idsf/TSNE_feature_data.npy', total_ids)
# np.save(f'/home/yl/lk/code/ID-disen_shows/ablation_shows/DIV_WO/KFFA/data/TSNE_feature_data.npy', total_ids)
# metric_results = {
#     # 'privacy_metric': np.mean(privacy_metric_list),
#     'recovery_ssim': np.mean(rec_ssim_list),
#     'recovery_lpips': np.mean(rec_lpips_list),
#     'recovery_mse': np.mean(rec_mse_list),
#     'recovery_psnr': np.mean(rec_psnr_list),

#     'Face_detection': np.mean(Face_detection_list),
#     'Bounding_box_distance': np.mean(Bounding_box_distance_list),
#     'Landmark_distance': np.mean(Landmark_distance_list),
#     'arc_cos_similarity': np.mean(arc_cos_similarity_list),
#     'vgg_cos_similarity': np.mean(vgg_cos_similarity_list),
#     'casia_cos_similarity_list': np.mean(casia_cos_similarity_list),
#     'anon_mse': np.mean(anon_mse_list),
#     'anon_mae': np.mean(anon_mae_list),
#     'arc_anon_mse': np.mean(arc_anon_mse_list),
#     'arc_anon_mae': np.mean(arc_anon_mae_list),
#     'vgg_anon_mse': np.mean(vgg_anon_mse_list),
#     'vgg_anon_mae': np.mean(vgg_anon_mae_list),

#     'fake_ssim': np.mean(fake_ssim_list),
#     'fake_mse': np.mean(fake_mse_list),
#     'fake_psnr': np.mean(fake_psnr_list),
#     'fake_lpips': np.mean(fake_lpips_list),


#     'Zfake_cos_similarity': np.mean(Zfake_cos_similarity_list),
#     'fake2_cos_similarity': np.mean(fake2_cos_similarity_list),
#     'Zrecon_cos_similarity': np.mean(Zrecon_cos_similarity_list),
#     'fakeZ_cos_similarity': np.mean(fakeZ_cos_similarity_list),


    
# }

metric_results = {
    # 'privacy_metric': np.mean(privacy_metric_list),
    'recovery_ssim': np.mean(rec_ssim_list),
    'recovery_lpips': np.mean(rec_lpips_list),
    'recovery_mse': np.mean(rec_mse_list),
    'recovery_psnr': np.mean(rec_psnr_list),
    'vgg_cos_similarity': np.mean(vgg_cos_similarity_list),
    'arc_cos_similarity': np.mean(arc_cos_similarity_list),
    'ada_cos_similarity': np.mean(ada_cos_similarity_list),
    'vgg_anon_mse': np.mean(vgg_anon_mse_list),
    'vgg_anon_mae': np.mean(vgg_anon_mae_list),
    'arc_anon_mse': np.mean(arc_anon_mse_list),
    'arc_anon_mae': np.mean(arc_anon_mae_list),
    'ada_anon_mae': np.mean(ada_anon_mae_list),
    'Face_detection': np.mean(Face_detection_list),
    'Bounding_box_distance': np.mean(Bounding_box_distance_list),
    'Landmark_distance': np.mean(Landmark_distance_list),
    'recon_arc_cos_similarity': np.mean(recon_arc_cos_similarity_list),

    
}

# print(f"****** {dname} ******")
print(metric_results)

