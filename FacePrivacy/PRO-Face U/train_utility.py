
from embedder import *

from utils.utils_func import *
import shutil

# import modules.Unet_common as common

# net = Model()
# # net.cuda()
# # init_model(net)
# # net = torch.nn.DataParallel(net, device_ids=c.device_ids)
# para = get_parameter_number(net)
# print(para)
from train_attr_classifier import AttrClassifierHead, get_celeba_attr_labels
import os

import argparse
import numpy as np
import torch
import logging
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms, datasets
from torchvision.transforms.functional import InterpolationMode
# from face_embedder import PrivFaceEmbedder
from face.face_recognizer import get_recognizer
from utils.utils_mlp_train import pass_epoch_utility
from utils.image_processing import Obfuscator, input_trans, input_trans_nonface, rgba_image_loader
from dataset.triplet_dataset import TripletDataset
# from evaluations import main as run_evaluation
# from evaluations import prepare_eval_data, run_eval
import config.config as c
# import config.config_blur as c

import logging
import sys
sys.path.append(os.path.join(c.DIR_PROJECT, 'SimSwap'))

from utils import utils_log

DIR_HOME = os.path.expanduser("~")
DIR_THIS_PROJECT = os.path.dirname(os.path.realpath(__file__))
DIR_PROJ = os.path.dirname(os.path.realpath(__file__))
print("Hello")
device = c.DEVICE


def prepare_logger(session):
    #### Create SummaryWriter
    import socket
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(f'{DIR_THIS_PROJECT}/runs/{current_time}_{socket.gethostname()}_{session}')
    writer = SummaryWriter(log_dir=log_dir)
    writer.iteration, writer.interval = 0, 10

    # Create logger
    log_file = os.path.join(log_dir, 'training.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='')

    ## Create directories to save generated images and models
    dir_train_out = os.path.join(log_dir, 'train_out')
    dir_checkpoints = os.path.join(log_dir, 'checkpoints')
    dir_eval_out = os.path.join(log_dir, 'eval_out')
    if not os.path.isdir(dir_train_out):
        os.makedirs(dir_train_out, exist_ok=True)
    if not os.path.isdir(dir_checkpoints):
        os.makedirs(dir_checkpoints, exist_ok=True)
    if not os.path.isdir(dir_eval_out):
        os.makedirs(dir_eval_out, exist_ok=True)

    ## Copy the config file to logdir
    shutil.copy('config/config.py', log_dir)

    return writer, dir_train_out, dir_checkpoints, dir_eval_out









if __name__ == '__main__':
    main(
        c.recognizer,
        c.obfuscator,
        c.utility_level,
        c.attr_rec_model,
        c.dataset_dir,
        c.eval_dir,
        c.eval_pairs,
        c.debug
    )
