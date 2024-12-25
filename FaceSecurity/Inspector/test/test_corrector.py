import yaml
import argparse

# from trainer.exp_mgpu_trainer_mse import ExpMultiGpuTrainer

import torch
import torch.nn as nn
import cv2
import os
import yaml
import numpy as np
import sewar
import torchattacks
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
    
    parser.add_argument('--detector', default='xception', type=str)
    parser.add_argument('--load_path', default='./detector_weight', type=str)
    
    parser.add_argument('--corrector', default='resnet18', type=str)
    parser.add_argument('--save_path', default='./corrector_weight', type=str)
    
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
    dataset_test = FaceForensics(config_test)
    dataloader_test = data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8)

    # train
    lpth = args.load_path
    spth = args.save_path

    acc = AccMeter()
    acc_max = 0
    
    
    # load loss and optim and detector model
    NAME_DEC = args.detector
    NAME_DATA = args.dataset
    detector, *_ = model_selection(modelname=NAME_DEC, num_out_classes=2)
    detector_pth = os.path.join(lpth, NAME_DEC, NAME_DATA, 'best.pth')
    detector.load_state_dict(torch.load(detector_pth))
    detector = torch.nn.SyncBatchNorm.convert_sync_batchnorm(detector).cuda()
    detector.cuda(gpu)
    detector = torch.nn.parallel.DistributedDataParallel(detector, device_ids=[gpu])
    detector.eval()
    for param in detector.parameters():
        param.requires_grad = False
    
    NAME_COR = args.corrector
    corrector, *_ = model_selection(modelname = NAME_COR, num_out_classes=2)
    corrector_pth = os.path.join(spth, NAME_COR, NAME_DEC, NAME_DATA, 'best.pth') 
    
    corrector.load_state_dict(torch.load(corrector_pth))
    corrector = torch.nn.SyncBatchNorm.convert_sync_batchnorm(corrector).cuda()
    corrector.cuda(gpu)
    corrector = torch.nn.parallel.DistributedDataParallel(corrector, device_ids=[gpu])
    
    atk = torchattacks.CW(detector, c=1, steps=40)
    atk.set_normalization_used(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    adv_fail = 0
    adv_success = 0
    
    corrector.eval()
    acc.reset()
    
    for i, _ in enumerate(tqdm(dataloader_test)):
        image, targets = _
        
        if torch.cuda.is_available():
            image = image.to(gpu)
            targets = targets.to(gpu)
        
        preds = detector(image)[0]
        
        if torch.argmax(torch.softmax(preds, dim = -1), dim = -1).long() == targets:
            adv_imges = atk(image, targets)
            adv_targets = torch.zeros_like(targets)
            
            adv_preds = detector(adv_imges)[0]
            
            if torch.argmax(torch.softmax(adv_preds, dim=-1), dim = -1).long() != targets:
                adv_targets[0] = 1
                adv_success += 1
            else:
                adv_targets[0] = 0
                adv_fail += 1
            
            adv_preds_softmax = torch.softmax(adv_preds, dim=-1)
            transition_weight = torch.sigmoid(corrector(adv_imges)[0]).unsqueeze(1)
            adv_suc_pred = transition_weight @ adv_preds_softmax.unsqueeze(-1)
            adv_suc_pred = adv_suc_pred.squeeze(-1).squeeze(-1)
            
            acc.update(adv_suc_pred, adv_targets, True)
                
    if gpu == 0:    
        acc_epoch = acc.mean_acc()
        
        print('fail num:%d, success num:%d' % (adv_fail, adv_success))
        print('ACC:%.4f, Best_ACC:%.4f' %(acc_epoch, acc_max))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    arg = arg_parser()
    config = arg.config
    
    if arg.dataset == 'FF++':
        config_path = "config/dataset/faceforensics.yml"
    elif arg.dataset == 'stargan':
        config_path = "config/dataset/stargan.yml"
    elif arg.dataset == 'stylegan':
        config_path = "config/dataset/stylegan.yml" 
        
    set_random_seed(arg.manual_seed, True)
    
    ngpus_per_node = torch.cuda.device_count()
    arg.ngpus_per_node = ngpus_per_node
    mp.spawn(main_worker, nprocs = ngpus_per_node, args = (arg, config, config_path))