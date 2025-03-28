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
def test_epoch_mm23(embedder, obfuscator,  utility_fc, noise_mk,recognizer, gender_classifier, dataloader,
                    swap_target_set=(), typeWR='', dir_image='./images'):

    pro_ssim_list = []
    pro_psnr_list = []
    pro_lpips_list = []
    pro_cos_list = []


    rec_ssim_list = []
    rec_psnr_list = []
    rec_lpips_list = []

    wrec_ssim_list = []
    wrec_psnr_list = []
    wrec_lpips_list = []

    features_list = []
    labels_list = []
    batch_acc_list = []
    face_list=[]
    batch_acc_listrev=[]

    swap_target_set_len = len(swap_target_set)
    # cartoon_set_len = len(cartoon_set)
    right=0
    # resize_transforms = [
    #     T.Resize(112, interpolation=InterpolationMode.BICUBIC),
    # ]
    sum0=0
    # recognizer.resize = T.Compose(resize_transforms)
    for i_batch, data_batch in tqdm(enumerate(dataloader)):
    # for i_batch, data_batch in enumerate(dataloader):
        i_batch += 1
        if i_batch>500:
            break
        xa,identity = data_batch
        # custom_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=0.5, std=0.5)
        # ])
        # image_path1 = '/home/lixiong/Projects/ProFaceUtility/one/1/Ann_Veneman_0001.jpg'
        # image_path2 = '/home/lixiong/Projects/ProFaceUtility/one/1/Ann_Veneman_0001.jpg'  # 替换为你的图像文件路径
        # image1 = Image.open(image_path1)
        # image2 = Image.open(image_path2)
        # tensor_image1 = custom_transform(image1)
        # tensor_image2 = custom_transform(image2)
        # tensor_image1 = torch.cat([tensor_image1.unsqueeze(0).to(device) for _ in range(1)], dim=0)
        # tensor_image2 = torch.cat([tensor_image2.unsqueeze(0).to(device) for _ in range(1)], dim=0)
        # embedding_orig1 = recognizer(recognizer.resize(tensor_image1))
        # embedding_adv1 = recognizer(recognizer.resize(tensor_image2))

        _bs, _c, _w, _h = xa.shape
        # _bs = 1
        xa = xa.to(device)
        # xa = Image.open('/home/yuanlin/Projects/ProFaceUtility/runs/Mar08_16-18-43_YL1_hybridAll_inv3_recTypeRandom_utility/train_out/Train_ep16_batch1_orig.jpg').convert("RGB")
        # custom_transform = transforms.Compose([
        #     transforms.Resize(112, interpolation=Image.BICUBIC),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=0.5, std=0.5)
        # ])
        # xa = custom_transform(xa)
        #
        # num_repeats = 1
        # xa = torch.cat([xa.unsqueeze(0).to("cuda:0") for _ in range(num_repeats)], dim=0)
        # xn = Image.open('/home/yuanlin/Projects/ProFaceUtility/runs/Mar05_20-30-03_YL1_hybridAll_inv3_recTypeRandom_utility/checkpoints/hybridAll_inv3_recTypeRandom_utility_x_ori1_iter3000.pth').convert("RGB")
        # custom_transform = transforms.Compose([
        #     transforms.Resize(112, interpolation=Image.BICUBIC),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=0.5, std=0.5)
        # ])
        # xn = custom_transform(xn)
        #
        #
        # xn = torch.cat([xn.unsqueeze(0).to("cuda:0") ], dim=0)
        # xa_adv = Image.open('/home/yuanlin/Projects/ProFaceUtility/runs/Mar08_16-18-43_YL1_hybridAll_inv3_recTypeRandom_utility/train_out/Train_ep16_batch1_adv.jpg').convert("RGB")
        # custom_transform = transforms.Compose([
        #     transforms.Resize(112, interpolation=Image.BICUBIC),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=0.5, std=0.5)
        # ])
        # xa_adv = custom_transform(xa_adv)
        #
        # num_repeats = 1
        # xa_adv = torch.cat([xa_adv.unsqueeze(0).to("cuda:0") for _ in range(num_repeats)], dim=0)

        # xn=torch.full((1, 3, 112, 112), 0.5).to(device)
        # image_size = (3, 112, 112)
        #
        # std_dev = 1
        #
        # gaussian_noise = np.random.normal(0, std_dev, image_size)
        # gaussian_noise = np.clip(gaussian_noise, -1, 1)
        # gaussian_noise = torch.tensor(gaussian_noise, dtype=torch.float32)
        #
        # num_repeats = 1
        # xn = torch.cat([gaussian_noise.unsqueeze(0).to("cuda:0") for _ in range(num_repeats)], dim=0)

        # embedding_orig = recognizer(recognizer.resize(xa))
        # xn = noise_mk(embedding_orig).repeat(1, 4).reshape(_bs, 3, _w, _h)

        # targ_img = None
        # obf_name = obfuscator.func.__class__.__name__
        # if obf_name in ['FaceShifter', 'SimSwap']:
        #     targ_img, _ = swap_target_set[i_batch % swap_target_set_len]
        #     # targ_img, _ = swap_target_set[i_batch]
        # elif obf_name == 'Mask':
        #     targ_img, _ = cartoon_set[(i_batch - 1) % cartoon_set_len]
        embedding_orig = recognizer(recognizer.resize(xa))
        xa_id = obfuscator.extract_features(xa).to(device)
        xa_obfs_id = noise_mk(xa_id).to(device)

        xa_obfs = obfuscator(xa, xa_obfs_id)
        xa_obfs.to(device)

################################################################################################################
        ## Create password from protection
        # password = random_password()
        password = 0
        skey1 = generate_key(password, _bs, _w, _h).to(device)
        skey1_dwt = dwt(skey1)

        password_a = dwt(torch.randint(0, 2, (_bs, 1, _w, _h)).mul(2).sub(1).to(device))
        utility_factor = 0
        utility_cond_init=c.utility_cond_init
        # utility_cond_init = torch.tensor[float(utility_factor), 1 - float(utility_factor)]).repeat(_bs, 1).to(device)
        utility_condition = utility_fc(utility_cond_init).repeat(1, 4).reshape(_bs, 1, _w // 2, _h // 2)
        # condition_utility = torch.full((_bs, 1, _w // 2, _h // 2), torch.tensor(utility_factor)).to(device)
        condition = torch.concat((password_a, utility_condition), dim=1)
       low1,low2,xa_out_z, xa_adv = embedder(xa,xa_obfs , condition)
        embedding_adv = recognizer(recognizer.resize(xa_adv))
        # cosine_sim_orig = torch.nn.functional.cosine_similarity(embedding_orig, embedding_obfs)
        # #cosine_sim_proc = torch.nn.functional.cosine_similarity(embedding_orig, embedding_proc).cpu()
        # #cosine_sim_expected = 1 - torch.nn.functional.relu(1 - cosine_sim_orig * np.power(5, utility_factor))
        # #print(f"Batch-{i_batch} Orig/Proc Cos:", float(cosine_sim_orig.mean()),  float(cosine_sim_proc.mean()))
        # face_acc=(acc(0.9,cos_loss(embedding_adv,embedding_orig)))
        # face_list.append(face_acc.detach().cpu())
        # gender_pred = gender_classifier(embedding_proc).cpu()

        # batch_acc = accuracy(gender_pred, gender_gt)
        # batch_acc_list.append(batch_acc.detach().cpu())

        # Correct recovery
        key_rec = skey1_dwt.repeat(1, 3, 1, 1) if c.SECRET_KEY_AS_NOISE else \
            gauss_noise((_bs, _c * 4, _w // 2, _h // 2)).to(device)
        xa_rev, xa_rev_2 = embedder(key_rec, xa_adv, condition, rev=True)  # Recovery using noisy image
        embedding_rev=recognizer(recognizer.resize(xa_rev))
        gender_predrev = gender_classifier(embedding_rev).cpu()
        # batch_accrev = accuracy(gender_predrev, gender_gt)

        # batch_acc_listrev.append(batch_accrev.detach().cpu())

        password = random_password()
        skey2 = generate_key(password, _bs, _w, _h).to(device)
        skey2_dwt = dwt(skey2)

        key_rec = skey2_dwt.repeat(1, 3, 1, 1) if c.SECRET_KEY_AS_NOISE else \
            gauss_noise((_bs, _c * 4, _w // 2, _h // 2)).to(device)
        xa_rev_wrong, xa_rev_wrong_2 = embedder(key_rec, xa_adv, condition, rev=True)

        if i_batch <= 500:
            save_image(normalize(xa), f"{dir_image}/batch{i_batch}_orig.jpg", nrow=4)
            save_image(normalize(xa_adv), f"{dir_image}/batch{i_batch}_adv.jpg", nrow=4)
            save_image(normalize(xa_obfs), f"{dir_image}/batch{i_batch}_obfs.jpg", nrow=4)
            # # save_image(normalize(img_z, True), f"{dir_image}/batch{i_batch}_{obf_name}_proc_byproduct.jpg", nrow=4)
            # save_image(normalize(xn), f"{dir_image}/batch{i_batch}_{obf_name}.jpg", nrow=4)
            # save_image(normalize(xa_rev), f"{dir_image}/batch{i_batch}_{obf_name}_rev_u.jpg", nrow=4)
            # # save_image(normalize(xa_rev_2, True), f"{dir_image}/batch{i_batch}_{obf_name}_rev_byproduct.jpg", nrow=4)
            # save_image(normalize(xa_rev_wrong, True), f"{dir_image}/batch{i_batch}_{obf_name}_rev_wrong_u.jpg", nrow=4)
            # save_image(normalize(xa_rev_wrong_2, True), f"{dir_image}/batch{i_batch}"
            #                                                   f"_{obf_name}_rev_wrong_byproduct.jpg", nrow=4)

            # ########### Recover the image using pre-obfuscated directly ###########
            # key_rec = skey1_dwt.repeat(1, 3, 1, 1) if c.SECRET_KEY_AS_NOISE else \
            #     gauss_noise((_bs, _c * 4, _w // 2, _h // 2)).to(device)
            # xa_revFromObfs, xa_revFromObfs_2 = embedder(key_rec, xa_obfs, skey1_dwt, rev=True)  # Recovery using noisy image
            # save_image(normalize(xa_revFromObfs, adaptive=True),
            #            f"{dir_image}/batch{i_batch}_{obf_name}_revFromObfs.jpg", nrow=4)
            # save_image(normalize(xa_revFromObfs_2, adaptive=True),
            #            f"{dir_image}/batch{i_batch}_{obf_name}_revFromObfs_byproduct.jpg", nrow=4)
            #
            # ########### Recovery image using wrong password with only 1 bit difference ###########
            # password = 1
            # skey_1bit = generate_key(password, _bs, _w, _h).to(device)
            # skey_1bit_dwt = dwt(skey_1bit)
            # key_rec = skey_1bit_dwt.repeat(1, 3, 1, 1) if c.SECRET_KEY_AS_NOISE else \
            #     gauss_noise((_bs, _c * 4, _w // 2, _h // 2)).to(device)
            # xa_rev_wrong1bit, xa_rev_wrong1bit_2 = embedder(key_rec, xa_proc, skey_1bit_dwt, rev=True)
            # save_image(normalize(xa_rev_wrong1bit, adaptive=True),
            #            f"{dir_image}/batch{i_batch}_{obf_name}_rev_wrong1bit.jpg", nrow=4)
            #
            # ########### Noise Test: Recovery with noised add on image ##########
            # for m in range(0, 11):
            #     key_rec = skey1_dwt.repeat(1, 3, 1, 1) if c.SECRET_KEY_AS_NOISE else \
            #         gauss_noise((_bs, _c * 4, _w // 2, _h // 2)).to(device)
            #     img_nose = gauss_noise(xa_proc.shape).to(device) * 0.001 * m
            #     xa_revNoise, xa_revNoise_2 = embedder(key_rec, xa_proc + img_nose, skey1_dwt, rev=True)  # Recovery using
            #     # noisy image
            #     save_image(normalize(xa_revNoise), f"{dir_image}/batch{i_batch}_{obf_name}_rev_noise{m}.jpg", nrow=4)
        sum0=sum0+1
        # diff = np.subtract(embedding_orig.cpu().detach().numpy(), embedding_adv.cpu().detach().numpy())  # 计算两组嵌入（embeddings1 和 embeddings2）之间的差异
        # dist = np.sum(np.square(diff), 1)
        embedding_orig = embedding_orig / torch.linalg.norm(embedding_orig, axis=1, keepdims=True)
        embedding_adv = embedding_adv / torch.linalg.norm(embedding_adv, axis=1, keepdims=True)
        dist = torch.nn.functional.cosine_similarity(embedding_adv, embedding_orig).detach().cpu().numpy().mean()
        # print(loss_cos)
        if dist<0.17:
            right=right+1


        xa_norm = normalize(xa)
         #### Privacy metrics
        xa_adv_norm = normalize(xa_adv)
        # SSIM
        pro_ssim_score = ssim(xa_norm, xa_adv_norm).detach().cpu()
        pro_ssim_list.append(pro_ssim_score)
        # PSNR
        pro_psnr = psnr(xa_norm, xa_adv_norm).detach().cpu()
        pro_psnr_list.append(pro_psnr)
        # LPIPS
        pro_lpips = lpips_loss(xa_norm, xa_adv_norm).detach().cpu()
        pro_lpips_list.append(pro_lpips)
        #
        pro_cos_list.append(dist)
        # #### Recovery metrics
        # # SSIM
        # xa_rev_norm = normalize(xa_rev)
        # rec_ssim_score = ssim(xa_rev_norm, xa_norm).detach().cpu()
        # rec_ssim_list.append(rec_ssim_score)
        # # PSNR
        # rec_psnr = psnr(xa_rev_norm, xa_norm).detach().cpu()
        # rec_psnr_list.append(rec_psnr)
        # # LPIPS
        # rec_lpips = lpips_loss(xa_rev_norm, xa_norm).detach().cpu()
        # rec_lpips_list.append(rec_lpips)
        #
        # #### Wrong Recovery metrics
        # xa_rev_wrong_norm = normalize(xa_rev_wrong, adaptive=True)
        #
        # if typeWR == 'RandWR':
        #     # SSIM
        #     wrec_ssim_score = ssim(xa_rev_wrong_norm, xa_norm).detach().cpu()
        #     wrec_ssim_list.append(wrec_ssim_score)
        #     # PSNR
        #     wrec_psnr = psnr(xa_rev_wrong_norm, xa_norm).detach().cpu()
        #     wrec_psnr_list.append(wrec_psnr)
        #     # LPIPS
        #     wrec_lpips = lpips_loss(xa_rev_wrong_norm, xa_norm).detach().cpu()
        #     wrec_lpips_list.append(wrec_lpips)
        # else:
        #     # SSIM
        #     wrec_ssim_score = ssim(xa_rev_wrong_norm, xa_obfs_norm).detach().cpu()
        #     wrec_ssim_list.append(wrec_ssim_score)
        #     # PSNR
        #     wrec_psnr = psnr(xa_rev_wrong_norm, xa_obfs_norm).detach().cpu()
        #     wrec_psnr_list.append(wrec_psnr)
        #     # LPIPS
        #     wrec_lpips = lpips_loss(xa_rev_wrong_norm, xa_obfs_norm).detach().cpu()
        #     wrec_lpips_list.append(wrec_lpips)


    # metric_results = {
    #     'pSSIM': float(np.mean(pro_ssim_list)),
    #     'pLPIPS': float(np.mean(pro_lpips_list)),
    #     'pPSNR': float(np.mean(pro_psnr_list)),
    #     'rSSIM': float(np.mean(rec_ssim_list)),
    #     'rLPIPS': float(np.mean(rec_lpips_list)),
    #     'rPSNR': float(np.mean(rec_psnr_list)),
    #     'wrSSIM': float(np.mean(wrec_ssim_list)),
    #     'wrLPIPS': float(np.mean(wrec_lpips_list)),
    #     'wrPSNR': float(np.mean(wrec_psnr_list)),
    # }
    #
    # return metric_results
    ssim_ori_adv=np.mean(pro_ssim_list)
    psnr_ori_adv = np.mean(pro_psnr_list)
    lpips_ori_adv=np.mean(pro_lpips_list)
    cos_ori_adv=np.mean(pro_cos_list)
    resule=right/sum0
    # print("Gender classify acc.", mean_acc)
    return ssim_ori_adv,psnr_ori_adv,lpips_ori_adv,cos_ori_adv,resule
def main(inv_nblocks, embedder_path, fc_path, noise_path,dataset_path, test_session_dir, typeWR, batch_size):

    workers = 0 if os.name == 'nt' else 8

    # Determine if an nvidia GPU is available
    print('Running on device: {}'.format(device))

    #### Define the models
    embedder = ModelDWT(n_blocks=inv_nblocks)
    state_dict = torch.load(embedder_path)
    embedder.load_state_dict(state_dict)
    embedder.to(device)

    utility_fc = UtilityConditioner()
    state_dict = torch.load(fc_path)
    utility_fc.load_state_dict(state_dict)
    utility_fc.to(device)

    noise_mk = Noisemaker()
    state_dict = torch.load(noise_path)
    noise_mk.load_state_dict(state_dict)
    noise_mk.to(device)

    fr_model1 = irse.MobileFaceNet(512)
    fr_model1.load_state_dict(torch.load('./Pretrained_FR_Models/mobile_face.pth'))
    fr_model1.to(device)
    fr_model1.eval()

    fr_model2 = irse.Backbone(50, 0.6, 'ir_se')
    fr_model2.load_state_dict(torch.load('./Pretrained_FR_Models/irse50.pth'))
    fr_model2.to(device)
    fr_model2.eval()

    fr_model = ir152.IR_152((112, 112))
    fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/ir152.pth'))
    fr_model.to(device)
    fr_model.eval()

    recognizer = get_recognizer(c.recognizer)
    recognizer.to(device)
    recognizer.eval()

    gender_classifier = AttrClassifierHead()
    state_dict = torch.load(os.path.join(DIR_PROJ, "face/gender_model/gender_classifier_AdaFaceIR100.pth"))
    gender_classifier.load_state_dict(state_dict)
    gender_classifier.to(device)
    gender_classifier.eval()

    print('Number of parameters: {}'.format(get_parameter_number(embedder)))

    # Target images used for face swapping
    test_frontal_set = datasets.ImageFolder(c.target_img_dir_test)
    test_frontal_nums = len(test_frontal_set)
    target_set_train_nums = int(test_frontal_nums * 0.9)
    target_set_test_nums = test_frontal_nums - target_set_train_nums
    torch.manual_seed(0)
    target_set_train, target_set_test = \
        torch.utils.data.random_split(test_frontal_set, [target_set_train_nums, target_set_test_nums])

    # Sticker images used for face masking in train and test
    # cartoon_set = datasets.ImageFolder(c.cartoon_face_path, loader=rgba_image_loader)
    # cartoon_num = len(cartoon_set)
    # _train_num = int(cartoon_num * 0.9)
    # _test_num = cartoon_num - _train_num
    # torch.manual_seed(1)
    # cartoon_set_train, cartoon_set_test = torch.utils.data.random_split(cartoon_set, [_train_num, _test_num])

    # Try run validation first
    embedder.eval()
    utility_fc.eval()
    noise_mk.eval()
    # Create obfuscator
    #obf_options = ['medianblur_15', 'blur_21_6_10', 'pixelate_9', 'faceshifter', 'simswap', 'mask']
    obf_options = ['simswap']




    # Create train dataloader
    dir_test = os.path.join(dataset_path)
    dataset_test = datasets.ImageFolder(dir_test, transform=input_trans)

    # # Add gender label to test set
    # celeba_attr_dict = get_celeba_attr_labels(attr_file=c.celeba_attr_file, attr='Male')
    # dataset_test.samples = [
    #     (p, (idx, celeba_attr_dict[os.path.basename(p)]))
    #     for p, idx in dataset_test.samples
    # ]

    loader_test = DataLoader(dataset_test, num_workers=workers, batch_size=batch_size, shuffle=False)

    results = {}
    for obf_opt in obf_options:
        print('__________ {} __________'.format(obf_opt))
        obfuscator = Obfuscator(obf_opt, device)
        obfuscator.eval()
        #utility_factors = np.arange(0, 1.1, 0.2)
        acc_list = []
        #for uf in utility_factors:
        ssim_orig_adv,psnr_orig_adv,lpips_orig_adv,cos,lc = test_epoch_mm23(
                embedder, obfuscator, utility_fc,noise_mk,recognizer, gender_classifier, loader_test,
                target_set_test, typeWR, dir_image=test_session_dir
            )
        # acc_list.append(obfuscator_metrics)
        print('ssim:{} - psnr:{} -lpips:{} -CosSim:{} -acc:{}'.format( ssim_orig_adv,psnr_orig_adv,lpips_orig_adv,cos,lc))

        # print('{}: {}'.format(obf_opt, obfuscator_metrics))
        # results[obf_opt] = obfuscator_metrics
        print('{}: {}'.format(obf_opt, acc_list))

    return results
if __name__ == '__main__':
    print("runs")
    embedder_configs = [
        [3, 'RandWR',
         os.path.join(DIR_PROJ, "/model/checkpoints/simswap_inv3_recTypeRandom_utility_ep23.pth"),
         os.path.join(DIR_PROJ, "/model/checkpoints/simswap_inv3_recTypeRandom_utility_utilityFC_ep23.pth"),
         os.path.join(DIR_PROJ, "/model/checkpoints/simswap_inv3_recTypeRandom_utility_ep3_iter500.pth"),
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
