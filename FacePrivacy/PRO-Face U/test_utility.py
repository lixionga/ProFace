from embedder import dwt, iwt, ModelDWT, UtilityConditioner,Noisemaker
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import Pretrained_FR_Models.irse as irse
from torchvision.utils import save_image
from torchvision.transforms.functional import InterpolationMode
from utils.utils_train import normalize, gauss_noise, accuracy
from utils.image_processing import Obfuscator, input_trans, rgba_image_loader
from torchmetrics import StructuralSimilarityIndexMeasure, MeanSquaredError, PeakSignalNoiseRatio
from utils.loss_functions import lpips_loss,cos_loss
from utils.utils_func import get_parameter_number
import config.config as c
import random, string
from PIL import Image
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA512
import Pretrained_FR_Models.irse as irse
import Pretrained_FR_Models.facenet as facenet
import Pretrained_FR_Models.ir152 as ir152
import sys
from torchvision import transforms as T
import torch
import time
import torch.nn as nn
import random
import numpy as np
import os
import math
from torch.autograd import Variable
from torchvision import transforms
import logging
import torchvision.transforms.functional as F
from torchvision.utils import save_image
from utils.loss_functions import  l1_loss, triplet_loss, lpips_loss, logits_loss,l2_loss,cos_loss
from torch.nn import TripletMarginWithDistanceLoss
import modules.Unet_common as common
from utils.image_processing import normalize, clamp_normalize
import config.config as c
import json
from tqdm import tqdm
from face.face_recognizer import get_recognizer
from train_attr_classifier import AttrClassifierHead, get_celeba_attr_labels
sys.path.append(os.path.join(c.DIR_PROJECT, 'SimSwap'))


DIR_HOME = os.path.expanduser("~")
DIR_PROJ = os.path.dirname(os.path.realpath(__file__))
DIR_EVAL_OUT = os.path.join(DIR_PROJ, 'eval_out')

print("Hello")
GPU0 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device =GPU0




def random_password(length=16):
   return ''.join(random.choice(string.printable) for i in range(length))

# 读取celebA测试数据集
def get_test_celeA(data_path, pairs_file):
    """
    :param data_path: 请给出图像的路径
    :param pairs_file: 请给出测试所需要的pairs.txt文件
    :return: 图像路径的列表和判断身份是否一致的布尔ndarray
    """
    img_list = []  # 创建一个存放图像的list
    issame = []  # 标记两组图像是否相同

    pairs_file_buf = open(pairs_file)  # 读取文件
    line = pairs_file_buf.readline()  # 跳过第一行 因为第一行是无关的内容
    line = pairs_file_buf.readline().strip()  # 读取一行，去除首尾空格
    while line:  # 只要文件有内容，就会读取
        line_strs = line.split('\t')  # 按空格(python制表符)分割
        if len(line_strs) == 3:  # 如果是3元素，则表示两张人脸是同一个人
            person_name = line_strs[0]  # 第一个元素是身份ID
            image_index1 = line_strs[1]  # 第二个元素是第一张图的索引
            image_index2 = line_strs[2]  # 第三个元素是第二张图的索引
            image_name1 = data_path + '/' + image_index1  # + '.jpg'  # 得到第一张人脸的地址
            image_name2 = data_path + '/' + image_index2  # + '.jpg'  # 得到第二张人脸的地址
            label = 1  # 标签为1表示是同一个身份
        elif len(line_strs) == 4:  # 表示两张人脸是不同的人
            person_name1 = line_strs[0]  # 第一个人的身份ID
            image_index1 = line_strs[1]  # 第一个人的索引
            person_name2 = line_strs[2]  # 第二个人的身份ID
            image_index2 = line_strs[3]  # 第二个人的索引
            image_name1 = data_path + '/' + image_index1  # + '.jpg'  # 得到第一张人脸的地址
            image_name2 = data_path + '/' + image_index2  # + '.jpg'  # 得到第二张人脸的地址
            label = 0  # 标签为0表示不同身份
        else:
            raise Exception('Line error: %s.' % line)

        # 分批存入
        img_list.append(image_name1)
        img_list.append(image_name2)
        if label == 1:
            issame.append(True)
        else:
            issame.append(False)

        line = pairs_file_buf.readline().strip()  # 读取下一行
    # 将list转换为ndarray
    issame = np.array(issame)
    return img_list, issame
def generate_key(password,  bs, w, h):
    '''
    Function to generate a secret key with length nbits, based on an input password
    :param password: string password
    :param bs, w, h: batch_size, weight, height
    :return: tensor of 1 and -1 in shape(bs, 1, w, h)
    '''
    salt = 1
    key = PBKDF2(password, salt, int(w * h / 8), count=10, hmac_hash_module=SHA512)
    list_int = list(key)
    array_uint8 = np.array(list_int, dtype=np.uint8)
    array_bits = np.unpackbits(array_uint8).astype(int) * 2 - 1
    array_bits_2d = array_bits.reshape((w, h))
    skey_tensor = torch.tensor(array_bits_2d).repeat(bs, 1, 1, 1)
    return skey_tensor


dwt.to(device)
lpips_loss.to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
mse = MeanSquaredError().to(device)
psnr = PeakSignalNoiseRatio().to(device)


#def acc(param, param1):
#    pass


def acc(threshold,predict):
    # 将余弦相似度张量与阈值比较，得到二元张量
    binary_predictions = (predict >= threshold).float()\
    # 计算准确率
    acc2 = binary_predictions.mean()
    return acc2
if __name__ == '__main__':
    print("runs")
    embedder_configs = [
        [3, 'RandWR',
         os.path.join(DIR_PROJ, "/home/lixiong/Projects/ProFaceUtility/runs/Dec29_23-14-01_YL1_simswap_inv3_recTypeRandom_utility/checkpoints/simswap_inv3_recTypeRandom_utility_ep23.pth"),
         os.path.join(DIR_PROJ, "/home/lixiong/Projects/ProFaceUtility/runs/Dec29_23-14-01_YL1_simswap_inv3_recTypeRandom_utility/checkpoints/simswap_inv3_recTypeRandom_utility_utilityFC_ep23.pth"),
         os.path.join(DIR_PROJ, "/home/lixiong/Projects/ProFaceUtility/runs/Dec25_02-00-59_YL1_simswap_inv3_recTypeRandom_utility/checkpoints/simswap_inv3_recTypeRandom_utility_ep3_iter500.pth"),
         ],

    ]

    # Path to original datasets
    datasets1k = (
        # ('CelebA', '/media/Data8T/Datasets/CelebA/align_crop_224/test'),
        # ('FFHQ', '/media/Data8T/Datasets/FFHQ128/test'),
        ('LFW', os.path.join(c.DIR_PROJECT, 'experiments/test_data/LFW')),
    )

    for inv_nblocks, typeWR, embedder_path, fc_path,noise_path in embedder_configs:
        print(f"******************* {inv_nblocks} inv blocks **********************")
        for dataset_name, dataset_path in datasets1k:
            test_session = f"{dataset_name}_{inv_nblocks}InvBlocks_{typeWR}_utility"
            test_session_dir = os.path.join(DIR_EVAL_OUT, test_session)
            os.makedirs(test_session_dir, exist_ok=True)
            result_file = os.path.join(DIR_EVAL_OUT, f"{test_session}.json")
            result_dict = main(inv_nblocks, embedder_path, fc_path, noise_path,dataset_path, test_session_dir, typeWR,
                               batch_size=1)
            # with open(result_file, 'w') as f:
            #     json.dump(result_dict, f)
