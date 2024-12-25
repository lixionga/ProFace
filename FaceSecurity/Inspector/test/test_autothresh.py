import yaml
import argparse


import torch
import torch.nn as nn
import cv2
import os
import yaml
import numpy as np
import random
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchattacks
from datetime import datetime
from torch.utils import data
from dataset.faceforensics import FaceForensics
from model.network.models import model_selection
from model.network.xception import Generator
from trainer.utils import AccMeter, AUCMeter
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
    parser.add_argument('--manual_seed', default=2024, type=int)
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:23459', type=str,
                        help='url used to set up distributed training')
    
    parser.add_argument('--batchsize', default=32, type=int,
                        help='world size for distributed training')
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--interval', default=5, type=int)
    
    parser.add_argument('--dataset', default='FF++', type=str)
    
    parser.add_argument('--model', default='xception', type=str)
    parser.add_argument('--autothresholder', default='fc', type=str)
    parser.add_argument('--save_path', default='./thresholder', type=str)
    parser.add_argument('--load_path_dec', default='./detector_weight', type=str)
    parser.add_argument('--load_path_rec', default='./recover_weight', type=str)
    
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


def torch2numpy(image):
    image = 0.5 + 0.5 * image
    image = image[0].detach().permute(1,2,0).cpu().numpy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


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
    spth = args.save_path
    lpth_dec = args.load_path_dec
    lpth_rec = args.load_path_rec

    acc = AccMeter()
    acc_adv = AccMeter()

    # load detector model
    NAME_DEC = args.model
    NAME_AUT = args.autothresholder
    NAME_DATA = args.dataset
    model, *_ = model_selection(modelname=NAME_DEC, num_out_classes=2)
    model_pth = os.path.join(lpth_dec, NAME_DEC, NAME_DATA, 'best.pth')
    model.load_state_dict(torch.load(model_pth, map_location=torch.device('cuda:'+str(gpu))))
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    model.cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    generator = Generator()
    generator_pth = os.path.join(lpth_rec, NAME_DEC, NAME_DATA, 'best.pth')
    generator.load_state_dict(torch.load(generator_pth, map_location = torch.device('cuda:'+str(gpu))))
    generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator).cuda()
    generator.cuda(gpu)
    generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[gpu])
    generator.eval()
    for param in generator.parameters():
        param.requires_grad = False


    autothresholder, *_ = model_selection(modelname = NAME_AUT, num_out_classes=2)
    autothresholder_pth = os.path.join(spth, NAME_AUT, NAME_DEC, NAME_DATA, 'best.pth')
    autothresholder.load_state_dict(torch.load(autothresholder_pth, map_location = torch.device('cuda:'+str(gpu))))
    autothresholder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(autothresholder).cuda()
    autothresholder.cuda(gpu)
    autothresholder = torch.nn.parallel.DistributedDataParallel(autothresholder, device_ids=[gpu])

    # attack initilization
    atk = torchattacks.FGSM(model, eps = 4/255)
    atk.set_normalization_used(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    autothresholder.eval()
    acc.reset()
    acc_adv.reset()
    mse = nn.MSELoss()
    mse_loss = 0
    cnt = 0
    
    for i, _ in enumerate(tqdm(dataloader_test)):
        image, targets = _
        
        if torch.cuda.is_available():
            image = image.to(gpu)
            targets = targets.to(gpu)
        
        adv_imges = atk(image, targets)
        
        with torch.no_grad():
            feat = model(image)[1]
            img = generator(feat, image)
            
            feat_adv = model(adv_imges)[1]
            adv_img = generator(feat_adv, image)
            
            pred = autothresholder(img)[0]
            pred_adv = autothresholder(adv_img)[0]
            label = torch.zeros_like(targets).float()
            
            acc.update(pred, label)
            acc_adv.update(pred_adv, 1 - label)

            mse_loss += mse(img, adv_img)
            cnt += 1
            
    if gpu == 0:
        acc_epoch = acc.mean_acc()
        acc_adv_epoch = acc_adv.mean_acc()
        
        acc_avg = (acc_epoch + acc_adv_epoch) / 2

        print('ACC:%.4f, ACC_ADV:%.4f, ACC_AVG:%.4f' % (acc_epoch, acc_adv_epoch, acc_avg))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    arg = arg_parser()
    config = arg.config
    set_random_seed(arg.manual_seed, True)
    
    ngpus_per_node = torch.cuda.device_count()
    arg.ngpus_per_node = ngpus_per_node
    
    if arg.dataset == 'FF++':
        config_path = "config/dataset/faceforensics.yml"
    elif arg.dataset == 'stargan':
        config_path = "config/dataset/stargan.yml"
    elif arg.dataset == 'stylegan':
        config_path = "config/dataset/stylegan.yml" 
    
    mp.spawn(main_worker, nprocs = ngpus_per_node, args = (arg, config, config_path))