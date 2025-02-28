# LFW evaluation
import argparse
import torch
import random
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import datasets
from tqdm import tqdm
import numpy as np
import embedder as eb
import os
from torchvision.utils import save_image
from utils.utils_eval import read_pairs, get_paths, evaluate
from face_embedder import PrivFaceEmbedder
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import logging
# from training_triplet_logits import get_batch_negative_index
from utils.utils_train import get_batch_negative_index

dir_home = os.path.expanduser("~")
dir_facenet = os.path.dirname(os.path.realpath(__file__))

from face.face_recognizer import get_recognizer
from utils.loss_functions import triplet_loss, lpips_loss
from torch.nn import TripletMarginWithDistanceLoss
from utils.image_processing import Obfuscator

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
lpips_loss.to(device)
perc_triplet_loss = TripletMarginWithDistanceLoss(distance_function=lambda x, y : lpips_loss(x, y), margin=0.5)


def normalize(x: torch.Tensor):
    x_norm = x.add(1.0).mul(0.5)
    return x_norm


def proc_for_rec(img_batch, zero_mean=False, resize=0, grayscale=False):
    _res = img_batch
    if zero_mean:
        _res = img_batch.sub(0.5).mul(2.0)
    if zero_mean:
        _res = img_batch.sub(0.5).mul(2.0)
    if resize and resize != img_batch.shape[-1]:
        _res = F.resize(_res, size=[resize, resize])
    if grayscale:
        _res = F.rgb_to_grayscale(_res)
        
    return _res
