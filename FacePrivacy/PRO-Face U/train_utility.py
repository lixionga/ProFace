
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


def main(rec_name, obf_options, utility_level, attr_rec_model, dataset_dir, eval_dir, eval_pairs, debug):

    batch_size = c.batch_size
    epochs = 50
    start_epoch, epoch_iter = 1, 0
    workers = 0 if os.name == 'nt' else 8
    max_batch = np.inf
    embedder_model_path = None

    if debug:
        epochs = 2
        max_batch = 10

    # Determine if an nvidia GPU is available
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    #### Define the models
    embedder = ModelDWT(n_blocks=c.INV_BLOCKS)
    embedder.to(device)
    init_model(embedder, device)
    # para = get_parameter_number(embedder)
    # print(para)

    utility_fc = UtilityConditioner()
    utility_fc.to(device)

    noise_mk = Noisemaker()
    state_dict = torch.load(os.path.join(DIR_PROJ, "/home/lixiong/Projects/ProFaceUtility/runs/Dec25_02-00-59_YL1_simswap_inv3_recTypeRandom_utility/checkpoints/simswap_inv3_recTypeRandom_utility_ep3_iter500.pth"))
    noise_mk.load_state_dict(state_dict)
    noise_mk.to(device)
    noise_mk.eval()

    params_trainable = (
        list(filter(lambda p: p.requires_grad, embedder.parameters()))
        + list(filter(lambda p: p.requires_grad, utility_fc.parameters()))
    )

    ### Define optimizer, scheduler, dataset, and dataloader
    optimizer = torch.optim.Adam(params_trainable, lr=c.lr, eps=1e-6, weight_decay=c.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, c.weight_step, gamma=c.gamma)

    recognizer = get_recognizer(rec_name)
    recognizer.to(device)
    recognizer.eval()

    gender_classifier = AttrClassifierHead()
    state_dict = torch.load(os.path.join(DIR_PROJ, "face/gender_model/gender_classifier_AdaFaceIR100.pth"))
    gender_classifier.load_state_dict(state_dict)
    gender_classifier.to(device)
    gender_classifier.eval()




    # #### Define the utility classification face_detection
    # classifier = None
    # if utility_level > 0 and attr_rec_model and os.path.isfile(attr_rec_model):
    #     classifier = FaceClassifierHead()
    #     _state_dict = torch.load(attr_rec_model)
    #     classifier.load_state_dict(_state_dict)
    #     classifier.to(device)
    #     classifier.eval()

    # Create obfuscator
    obfuscator = Obfuscator(obf_options, device)

    # Create train dataloader
    dir_train = os.path.join(dataset_dir, 'train')
    dataset_train = datasets.ImageFolder(dir_train, transform=input_trans)
    celeba_attr_dict = get_celeba_attr_labels(attr_file=c.celeba_attr_file, attr='Male')
    dataset_train.samples = [
        (p, (idx, celeba_attr_dict[os.path.basename(p)]))
        for p, idx in dataset_train.samples
    ]
    loader_train = DataLoader(dataset_train, num_workers=workers, batch_size=batch_size, shuffle=True)




    # Create valid dataloader
    dir_valid = os.path.join(dataset_dir, 'valid')

    dataset_valid = datasets.ImageFolder(dir_valid, transform=input_trans)
    dataset_valid.samples = [
        (p, (idx, celeba_attr_dict[os.path.basename(p)]))
        for p, idx in dataset_valid.samples
    ]
    loader_valid = DataLoader(dataset_valid, num_workers=workers, batch_size=batch_size, shuffle=True)






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
