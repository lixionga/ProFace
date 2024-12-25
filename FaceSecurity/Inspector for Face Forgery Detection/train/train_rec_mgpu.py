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
from datetime import datetime
from torch.utils import data
from dataset.faceforensics import FaceForensics
from model.network.models import model_selection
from model.network.xception import Generator

from scheduler import get_scheduler
from optimizer import get_optimizer
from tqdm import tqdm

def arg_parser():
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config",
                        type=str,
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
    parser.add_argument('--save_path', default='./recover_weight', type=str)
    parser.add_argument('--load_path', default='./detector_weight', type=str)
    
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
    dataset = FaceForensics(config)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank) #
    dataloader = data.DataLoader(dataset, batch_size=args.batchsize, shuffle=(train_sampler is None), num_workers=8, sampler=train_sampler)

    dataset_test = FaceForensics(config_test)
    dataloader_test = data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8)

    # train
    spth = args.save_path
    num_epoch = args.epoch
    date = datetime.now().strftime('%Y%m%d%H%M')
    lpth = args.load_path

    least_mse = 1e6
    
    # load detector model
    NAME_DEC = args.model
    NAME_DATA = args.dataset
    model, *_ = model_selection(modelname=NAME_DEC, num_out_classes=2)
    model_pth = os.path.join(lpth, NAME_DEC, NAME_DATA, 'best.pth')
    model.load_state_dict(torch.load(model_pth, map_location=torch.device('cuda:'+str(gpu))))
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    model.cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    generator = Generator()
    generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator).cuda()
    generator.cuda(gpu)
    generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[gpu])
    
    # load loss and optim
    mse = nn.MSELoss().to(gpu)
    optim = get_optimizer('adam')(generator.parameters(), lr = 0.0002, weight_decay = 0.00001)
    scheduler = get_scheduler(optim, {'name': "StepLR", 'step_size': 20, 'gamma': 0.1})
    
    for epoch in range(num_epoch):
        # train
        generator.train()
        mse.train()
        
        if gpu == 0:
            dataloader = tqdm(dataloader, position=0, leave=True, dynamic_ncols=True)
            
        for i, _ in enumerate(dataloader):
            image, targets = _
            
            if torch.cuda.is_available():
                image = image.to(gpu)
                targets = targets.to(gpu)
            
            feat = model(image)[1]
            gen_img = generator(feat, image)
            
            generator.zero_grad()
            loss = mse(gen_img, image)
            loss.backward()
            optim.step()

            if gpu == 0:
                current_lr = optim.state_dict()['param_groups'][0]['lr']
                dataloader.set_description(
                    'Epoch: {Epoch:d}|lr: {lr:.7f}|loss: {loss:.8f}'.format(
                        Epoch = epoch, 
                        loss = loss.item(),
                        lr = current_lr))
            
        # validation
        if gpu == 0 and (epoch+1) % args.interval == 0:
            generator.eval()
            mse_loss = 0
            
            with torch.no_grad():
                for i, _ in enumerate(tqdm(dataloader_test)):
                    image, targets = _
                    
                    if torch.cuda.is_available():
                        image = image.to(gpu)
                        targets = targets.to(gpu)
                        
                    feat = model(image)[1]
                    gen_img = generator(feat, image)
                    mse_loss += mse(gen_img, image)


                if mse_loss/dataloader_test.__len__() < least_mse:
                    least_mse = mse_loss/dataloader_test.__len__()
                    torch.save(generator.module.state_dict(), os.path.join(spth, NAME_DEC, NAME_DATA, 'best'+ date +'.pth'))
                    print('least_mse', least_mse)
                
        scheduler.step()


if __name__ == '__main__':
    arg = arg_parser()
    config = arg.config
    set_random_seed(arg.manual_seed, True)
    
    ngpus_per_node = torch.cuda.device_count()
    arg.ngpus_per_node = ngpus_per_node
    
    if arg.dataset == 'FF++':
        config_path = "data_config/faceforensics.yml"
    elif arg.dataset == 'stargan':
        config_path = "data_config/stargan.yml"
    elif arg.dataset == 'stylegan':
        config_path = "data_config/stylegan.yml" 
    
    mp.spawn(main_worker, nprocs = ngpus_per_node, args = (arg, config, config_path))