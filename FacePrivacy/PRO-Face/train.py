import numpy as np

from embedder import ProFaceEmbedder
import shutil
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms, datasets
from face.face_recognizer import get_recognizer
from utils.utils_train import pass_epoch
from utils.image_processing import Obfuscator, input_trans
from dataset.triplet_dataset import TripletDataset
import config.config as c

import logging
import sys
sys.path.append(os.path.join(c.PROJECT_DIR, 'SimSwap'))


DIR_HOME = os.path.expanduser("~")
DIR_THIS_PROJECT = os.path.dirname(os.path.realpath(__file__))


def main(rec_name, obf_options, dataset_dir, debug=False):

    batch_size = c.batch_size
    epochs = 50
    start_epoch, epoch_iter = 1, 0
    workers = 0 if os.name == 'nt' else 8
    max_batch = np.Inf

    if debug:
        epochs = 2
        max_batch = 10

    # Determine if an nvidia GPU is available
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    ### Define the models
    embedder = ProFaceEmbedder().to(device)

    ### Define optimizer
    optimizer = optim.Adam(embedder.parameters(), lr=0.001)
    scheduler = None #MultiStepLR(optimizer, [5, 10])

    recognizer = get_recognizer(rec_name)
    recognizer.to(device)
    recognizer.eval()

    obfuscator = Obfuscator(obf_options)

    # Create train dataloader
    dir_train = os.path.join(dataset_dir, 'train')
    filelist_train = os.path.join(dataset_dir, 'filenames_train.txt')
    triplets_train = os.path.join(dataset_dir, 'triplets_train.txt')
    dataset_train = TripletDataset(dir_train, filelist_train, triplets_train, transform=input_trans)
    train_loader = DataLoader(dataset_train, num_workers=workers, batch_size=batch_size, shuffle=True)

    target_set_train = []
    if obfuscator.name in ['faceshifter', 'simswap']:
        # Create dataloder for target images for face swapping
        target_dir_train = os.path.join(dataset_dir, 'valid_frontal')
        target_dir_test = os.path.join(dataset_dir, 'test_frontal')
        target_set_train = datasets.ImageFolder(target_dir_train, transform=obfuscator.targ_img_trans)


    # Create valid dataloader
    dir_valid = os.path.join(dataset_dir, 'valid')
    filelist_valid = os.path.join(dataset_dir, 'filenames_valid.txt')
    triplets_valid = os.path.join(dataset_dir, 'triplets_valid.txt')
    dataset_valid = TripletDataset(dir_valid, filelist_valid, triplets_valid, transform=input_trans)
    valid_loader = DataLoader(dataset_valid, num_workers=workers, batch_size=batch_size, shuffle=True)

    #### Create SummaryWriter
    import socket
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    session = f'{obf_options}_{rec_name}'
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

    #### Train model
    print('\n-------------- Start training ----------------')
    for epoch in range(start_epoch, epochs + start_epoch):
        logging.info('\nEpoch {}/{}'.format(epoch, epochs))
        logging.info('-' * 11)
        print('\nEpoch {}/{}'.format(epoch, epochs))
        print('-' * 11)

        embedder.train()
        obfuscator.train()
        _, _, models_saved = pass_epoch(
            embedder, recognizer, obfuscator, train_loader, target_set_train, session=session,
            dir_image=dir_train_out,
            dir_checkpoint=dir_checkpoints, optimizer=optimizer, scheduler=scheduler, show_running=True, device=device,
            writer=writer, epoch=epoch, max_batch=max_batch, logger_train=None
        )

        embedder.eval()
        obfuscator.eval()
        pass_epoch(
            embedder, recognizer, obfuscator, valid_loader, target_set_train, session=session,
            dir_image=dir_train_out, optimizer=optimizer, scheduler=scheduler, show_running=True, device=device,
            writer=writer, epoch=epoch, max_batch=max_batch
        )

        model_name = f'{session}_ep{epoch}'
        saved_path = f'{dir_checkpoints}/{model_name}.pth'
        torch.save(embedder.state_dict(), saved_path)

    print('-------------- Training done ----------------')


if __name__ == '__main__':
    main(
        c.recognizer,
        c.obfuscator,
        c.dataset_dir,
        c.debug
    )

