
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

    noise_mk = Noisemaker()
    noise_mk.to(device)

    params_trainable = (
         list(filter(lambda p: p.requires_grad, noise_mk.parameters()))
    )

    ### Define optimizer, scheduler, dataset, and dataloader
    optimizer = torch.optim.Adam(params_trainable, lr=c.lr, eps=1e-6, weight_decay=c.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, c.weight_step, gamma=c.gamma)

    recognizer = get_recognizer(rec_name)
    recognizer.to(device)
    recognizer.eval()





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

    # loader_nonface_train, loader_nonface_valid = None, None
    # if utility_level == c.Utility.FACE:
    #     dataset_nonface = datasets.ImageFolder(c.mini_imagenet_dir, transform=input_trans_nonface)
    #     dataset_nonface.samples = [(p, (id_label, 1)) for p, id_label in dataset_nonface.samples] # Label nonface as 1
    #     _dataset_nonface_size = len(dataset_nonface)
    #     _train_size = int(_dataset_nonface_size * 0.7)
    #     _valid_size = int(_dataset_nonface_size * 0.2)
    #     _test_size  = _dataset_nonface_size - _train_size - _valid_size
    #     dataset_nonface_train, dataset_nonface_valid, dataset_nonface_test = random_split(
    #         dataset=dataset_nonface,
    #         lengths=[_train_size, _valid_size, _test_size],
    #         generator=torch.Generator().manual_seed(0)
    #     )
    #
    #     loader_nonface_train = DataLoader(
    #         dataset_nonface_train,
    #         num_workers=workers,
    #         batch_size=batch_size * 3,
    #         shuffle=True
    #     )
    #
    #     loader_nonface_valid = DataLoader(
    #         dataset_nonface_valid,
    #         num_workers=workers,
    #         batch_size=batch_size * 3,
    #         shuffle=True
    #     )

    # # # Create evaluation dataloader
    # test_loader, test_path_list, test_issame_list = prepare_eval_data(eval_dir, eval_pairs, input_trans)

    # Target images used for face swapping in train and test
    target_set_train, target_set_test = [], []

    # Sticker images used for face masking in train and test
    cartoon_set_train, cartoon_set_test = [], []

    if obfuscator.name in ['faceshifter', 'simswap', 'hybridMorph', 'hybridAll']:
        # target_dir_train = os.path.join(dataset_dir, 'valid_frontal')
        # target_dir_test = os.path.join(dataset_dir, 'test_frontal')
        # target_set_train = datasets.ImageFolder(target_dir_train)
        test_frontal_set = datasets.ImageFolder(os.path.join(dataset_dir, 'test_frontal'))
        test_frontal_nums = len(test_frontal_set)
        target_set_train_nums = int(test_frontal_nums * 0.9)
        target_set_test_nums  = test_frontal_nums - target_set_train_nums
        torch.manual_seed(0)
        target_set_train, target_set_test = \
            torch.utils.data.random_split(test_frontal_set, [target_set_train_nums, target_set_test_nums])

    if obfuscator.name in ['mask', 'hybridAll']:
        cartoon_set = datasets.ImageFolder(c.cartoon_face_path, loader=rgba_image_loader)
        cartoon_num = len(cartoon_set)
        _train_num = int(cartoon_num * 0.9)
        _test_num  = cartoon_num - _train_num
        torch.manual_seed(1)
        cartoon_set_train, cartoon_set_test = torch.utils.data.random_split(cartoon_set, [_train_num, _test_num])

    suffix = '_utility' if c.SECRET_KEY_AS_NOISE else ''
    session = f'{obf_options}_inv{c.INV_BLOCKS}_recType{c.WRONG_RECOVER_TYPE}{suffix}'
    writer, dir_train_out, dir_checkpoints, dir_eval_out = prepare_logger(session)

    #### Train face_detection
    print('\n-------------- Start training ----------------')

    # Try run validation first
    embedder.eval()
    obfuscator.eval()
    noise_mk.eval()
    with torch.no_grad():
        pass_epoch_utility(
        embedder,noise_mk, obfuscator, recognizer, loader_valid, target_set_test,
        cartoon_set_test,
        dir_image=dir_train_out, optimizer=optimizer, scheduler=scheduler, show_running=True, writer=writer,
        epoch=0, debug=True
    )

    for epoch in range(start_epoch, epochs + start_epoch):
        logging.info('\nEpoch {}/{}'.format(epoch, epochs))
        logging.info('-' * 11)
        print('\nEpoch {}/{}'.format(epoch, epochs))
        print('-' * 11)

        embedder.train()
        obfuscator.train()
        noise_mk.train()
        _, _, models_saved = pass_epoch_utility(
            embedder, noise_mk,obfuscator, recognizer, loader_train, target_set_train, cartoon_set_train,
            session=session, dir_image=dir_train_out, dir_checkpoint=dir_checkpoints, optimizer=optimizer,
            scheduler=None, show_running=True, writer=writer, epoch=epoch, debug=debug
        )

        ###embedder.eval()
        #obfuscator.eval()
        #utility_fc.eval()
        #with torch.no_grad():
            #pass_epoch_utility(
            #embedder, utility_fc, obfuscator, gender_classifier,recognizer, loader_valid, target_set_test, cartoon_set_test,
            #dir_image=dir_train_out, optimizer=optimizer, scheduler=scheduler, show_running=True, writer=writer,
            #epoch=epoch, debug=debug
        #)

        model_name = f'{session}_noise_mk_ep{epoch}'
        saved_path = f'{dir_checkpoints}/{model_name}.pth'
        torch.save(noise_mk.state_dict(), saved_path)


        # ## Run testing
        # logging.info(f'-------------- Evaluation ----------------')
        # print(f'-------------- Evaluation ----------------')
        # embedder.eval()
        # obfuscator.eval()
        # run_eval(embedder, recognizer, obfuscator, test_loader, test_path_list, test_issame_list,
        #          [], dir_eval_out, model_name)

    print('-------------- Done ----------------')



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

