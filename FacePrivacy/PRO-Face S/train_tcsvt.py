import os
import numpy as np
import torch
import shutil
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from utils.utils_train import pass_epoch_mm23
from utils.image_processing import Obfuscator, input_trans, rgba_image_loader
from embedder import *
from utils.utils_func import *
import config.config as c
from dataset.CelebA import CelebAImageDataset

import logging
import sys
sys.path.append(os.path.join(c.DIR_PROJECT, 'SimSwap'))


DIR_HOME = os.path.expanduser("~")
DIR_THIS_PROJECT = os.path.dirname(os.path.realpath(__file__))

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
    DIR_TMP_config = os.path.join(c.DIR_PROJECT, 'config/config.py')
    DIR_TMP_util_train = os.path.join(c.DIR_PROJECT, 'utils/utils_train.py')
    DIR_TMP_loss_functions = os.path.join(c.DIR_PROJECT, 'utils/loss_functions.py')
    shutil.copy(DIR_TMP_config, log_dir)
    shutil.copy(DIR_TMP_util_train, log_dir)
    shutil.copy(DIR_TMP_loss_functions, log_dir)

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
    print('Running on device: {}'.format(device))

    #### Define the models
    embedder = ModelDWT(n_blocks=c.INV_BLOCKS)
    embedder.to(device)
    init_model(embedder, device)
    para = get_parameter_number(embedder)
    print(para)
    params_trainable = (list(filter(lambda p: p.requires_grad, embedder.parameters())))

    ### Define optimizer, scheduler, dataset, and dataloader
    optimizer = torch.optim.Adam(params_trainable, lr=c.lr, eps=1e-6, weight_decay=c.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, c.weight_step, gamma=c.gamma)

    # Create obfuscator
    obfuscator = Obfuscator(obf_options, device)

    # Create train dataloader
    dir_train = os.path.join(dataset_dir, 'train')
    dataset_train = CelebAImageDataset(dir_train, transform=input_trans)
    loader_train = DataLoader(dataset_train, num_workers=workers, batch_size=batch_size, shuffle=True, drop_last=True)

    # Create valid dataloader
    dir_valid = os.path.join(dataset_dir, 'test')
    dataset_valid = CelebAImageDataset(dir_valid, transform=input_trans)
    loader_valid = DataLoader(dataset_valid, num_workers=workers, batch_size=batch_size, shuffle=True, drop_last=True)

    # Target images used for face swapping in train and test
    target_set_train, target_set_test = [], []

    # Sticker images used for face masking in train and test
    cartoon_set_train, cartoon_set_test = [], []
    if obfuscator.name in ['faceshifter', 'simswap', 'hybridMorph', 'hybridAll']:
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

    suffix = '_secretAsNoise_TripMargin1.2' if c.SECRET_KEY_AS_NOISE else ''
    session = f'{obf_options}_inv{c.INV_BLOCKS}_recType{c.WRONG_RECOVER_TYPE}{suffix}'
    writer, dir_train_out, dir_checkpoints, dir_eval_out = prepare_logger(session)

    #### Train face_detection
    print('\n-------------- Start training ----------------')

    # Try run validation first
    embedder.eval()
    obfuscator.eval()
    pass_epoch_mm23(
        embedder, obfuscator, loader_valid, target_set_test,
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
        _, _, models_saved = pass_epoch_mm23(
            embedder, obfuscator, loader_train, target_set_train, cartoon_set_train,
            session=session, dir_image=dir_train_out, dir_checkpoint=dir_checkpoints, optimizer=optimizer,
            scheduler=None, show_running=True, writer=writer, epoch=epoch, debug=debug
        )

        embedder.eval()
        obfuscator.eval()
        pass_epoch_mm23(
            embedder, obfuscator, loader_valid, target_set_test, cartoon_set_test,
            dir_image=dir_train_out, optimizer=optimizer, scheduler=scheduler, show_running=True, writer=writer,
            epoch=epoch, debug=debug
        )

        model_name = f'{session}_ep{epoch}'
        saved_path = f'{dir_checkpoints}/{model_name}.pth'
        torch.save(embedder.state_dict(), saved_path)

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