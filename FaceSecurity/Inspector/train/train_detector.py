import yaml
import argparse

# from trainer.exp_mgpu_trainer_mse import ExpMultiGpuTrainer

import torch
import torch.nn as nn
import cv2
import os
import yaml
import numpy as np
import random
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime
from torch.utils import data
from dataset.faceforensics import FaceForensics
from model.network.models import model_selection

from scheduler import get_scheduler
from optimizer import get_optimizer
from trainer.utils import AccMeter, AUCMeter
from tensorboardX import SummaryWriter
from tqdm import tqdm

def arg_parser():
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config",
                        type=str,
                        default="config/Recce.yml",
                        help="Specified the path of configuration file to be used.")
    parser.add_argument("--local_rank", default=-1,
                        type=int,
                        help="Specified the node rank for distributed training.")
    parser.add_argument('--world_size', default=1, type=int,
                        help='world size for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--manual_seed', default=2023, type=int) 
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:23459', type=str,
                        help='url used to set up distributed training')
    
    parser.add_argument('--batchsize', default=32, type=int,
                        help='world size for distributed training')
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--interval', default=5, type=int)
    
    parser.add_argument('--dataset', default='FF++', type=str)
    parser.add_argument('--model', default='xception', type=str)
    parser.add_argument('--save_path', default='/detector_weight', type=str)
    
    return parser.parse_args()


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main_worker(gpu, args, cfg, config_path):

    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        
    with open(config_path) as config_file:
        config_test = yaml.load(config_file, Loader=yaml.FullLoader)

    config = config["train_cfg"]
    config_test = config_test["test_cfg"]
    
    # local rank allocation 
    args.rank = args.rank * args.ngpus_per_node + gpu
    
    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    torch.cuda.set_device(gpu)
    print(f"{args.dist_url}, ws:{args.world_size}, rank:{args.rank}")
    
    # load dataset
    dataset = FaceForensics(config)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank) #
    dataloader = data.DataLoader(dataset, batch_size=args.batchsize, shuffle=(train_sampler is None), num_workers=8, sampler=train_sampler)

    dataset_test = FaceForensics(config_test)
    dataloader_test = data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8)

    # train
    spth = args.save_path
    num_epoch = args.epoch
    date = datetime.now().strftime('%Y%m%d%H%M')

    acc = AccMeter()
    auc = AUCMeter()
    acc_max = 0
    auc_max = 0
    iteration = 0
    
    # load loss and optim and detector model
    NAME = args.model
    NAME_DATA = args.dataset
    model, *_ = model_selection(modelname=NAME, num_out_classes=2) # softmax sigmoid
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    model.cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    print('Detector: %s' %(NAME))
    
    ce = nn.CrossEntropyLoss().to(gpu)
    
    optim = get_optimizer('adam')(model.parameters(), lr = 0.0002, weight_decay= 0.00001)
    scheduler = get_scheduler(optim, {'name': "StepLR", 'step_size': 20, 'gamma': 0.1})
    
    for epoch in range(num_epoch):
        # train
        model.train()
        ce.train()
        
        if gpu == 0:
            dataloader = tqdm(dataloader, position=0, leave=True, dynamic_ncols=True)
            
        for i, _ in enumerate(dataloader):
            image, targets, path, texts, muti_label = _
            
            if torch.cuda.is_available():
                image = image.to(gpu)
                # targets = targets.float()[:, None]
                targets = targets.to(gpu)
                
            pred = model(image)

            optim.zero_grad()
            loss = ce(pred, targets)
            loss.backward()
            optim.step()

            if gpu == 0:
                current_lr = optim.state_dict()['param_groups'][0]['lr']
                dataloader.set_description(
                    'Epoch: {Epoch:d}|lr: {lr:.7f}|loss: {loss:.8f}'.format(
                        Epoch = epoch, 
                        loss = loss.item(),
                        lr = current_lr))
            # writer.add_scalar('Loss', loss.item(), epoch * iteration + i)
            
        # validation
        if gpu == 0 and (epoch+1) % args.interval == 0:
            model.eval()
            acc.reset()
            
            with torch.no_grad():
                for i, _ in enumerate(tqdm(dataloader_test)):
                    image, targets, path, texts, muti_label = _
                    
                    if torch.cuda.is_available():
                        image = image.to(gpu)
                        targets = targets.to(gpu)
                        
                    pred = model(image)
                    acc.update(pred, targets)
                
                acc_epoch = acc.mean_acc()
                
                if acc_epoch > acc_max:
                    acc_max = acc_epoch
                    torch.save(model.module.state_dict(), os.path.join(spth, NAME, NAME_DATA, 'best_'+ date +'.pth'))

                print('ACC:%.4f, Best_ACC:%.4f' %(acc_epoch, acc_max))
            
        scheduler.step()


if __name__ == '__main__':
    arg = arg_parser()
    config = arg.config
    if arg.dataset == 'FF++':
        config_path = "config/dataset/faceforensics.yml"
    elif arg.dataset == 'stargan':
        config_path = "config/dataset/stargan2.yml"
    elif arg.dataset == 'stylegan':
        config_path = "config/dataset/stylegan2.yml" 
    set_random_seed(arg.manual_seed, True)
    
    ngpus_per_node = torch.cuda.device_count()
    arg.ngpus_per_node = ngpus_per_node
    mp.spawn(main_worker, nprocs = ngpus_per_node, args = (arg, config, config_path))