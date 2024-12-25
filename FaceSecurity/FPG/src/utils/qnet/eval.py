import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as opti
from tqdm import tqdm
import torchvision.transforms as T
from .generate_pseudo_labels.extract_embedding.model import model
import numpy as np
from scipy import stats
import pdb
from PIL import Image
import cv2

def read_img(imgPath):
    data = torch.randn(1, 3, 112, 112)
    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = Image.open(imgPath).convert("RGB")
    data[0, :, :, :] = transform(img)
    return data


def read_img1(imgPath):
    data = torch.randn(1, 3, 112, 112)
    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = cv2.imread(imgPath)
    img = torch.Tensor(img).permute(2,0,1)
    img = img.permute(1,2,0)
    img = img.numpy()
    img = img.astype(np.uint8)
    
    img = Image.fromarray(img)
    data[0, :, :, :] = transform(img)
    print(data.shape)
    return data


def read_img2(imgPath):
    data = torch.randn(1, 3, 112, 112)
    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[760:1140,:380]
    img = torch.Tensor(img).permute(2,0,1)
    img = img.permute(1,2,0)
    img = img.numpy()
    img = img.astype(np.uint8)
    print('---', img[:10,:10,0])
    
    img = Image.fromarray(img).convert("RGB")
    data[0, :, :, :] = transform(img)
    
    print('0',data[0,0,:10,:10])
    
    return data


def read_imgs2(img):     # read image & data pre-process
    transform = T.Resize((112, 112), antialias=True)
    normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    data = img * 255
    data = data.type(torch.uint8)
    data = transform(data)
    default_float_dtype = torch.get_default_dtype()
    data = data.type(default_float_dtype).div(255)
    data = normalize(data)
    
    return data

def read_imgs(img, gpu):     # read image & data pre-process
    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    data = torch.zeros(img.shape[0], 3, 112, 112)
    
    for i in range(img.shape[0]):
        tmp = img[i].cpu().permute(1,2,0) * 255
        tmp = tmp.numpy()
        tmp = tmp.astype(np.uint8)
        tmp = Image.fromarray(tmp).convert("RGB")

        data[i, :, :, :] = transform(tmp)
        
    return data.to(gpu)


def load_quality(qnet, img, div = 50.):
    data = read_imgs2(img)
    quality = qnet(data)

    quality = quality / div
    
    return quality

    
def network(eval_model, device):
    net = model.R50([112, 112], use_type="Qua").to(device)
    net_dict = net.state_dict()     
    data_dict = {
        key.replace('module.', ''): value for key, value in torch.load(eval_model, map_location=device).items()}
    net_dict.update(data_dict)
    net.load_state_dict(net_dict)
    net.eval()
    
    return net
    