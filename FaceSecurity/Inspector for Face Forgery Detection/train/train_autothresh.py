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
from scheduler import get_scheduler
from optimizer import get_optimizer
from trainer.utils import AccMeter
from tqdm import tqdm
from apex import amp

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
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--interval', default=2, type=int)
    
    parser.add_argument('--dataset', default='FF++', type=str)
    
    parser.add_argument('--model', default='xception', type=str)
    parser.add_argument('--autothresholder', default='resnet18', type=str)
    
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
    dataset = FaceForensics(config)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank) #
    dataloader = data.DataLoader(dataset, batch_size=args.batchsize, shuffle=(train_sampler is None), num_workers=8, sampler=train_sampler)

    dataset_test = FaceForensics(config_test)
    dataloader_test = data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8)

    # train
    spth = args.save_path
    num_epoch = args.epoch
    date = datetime.now().strftime('%Y%m%d%H%M')
    lpth_dec = args.load_path_dec
    lpth_rec = args.load_path_rec

    acc = AccMeter()
    acc_adv = AccMeter()
    acc_save = 0
    
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
    generator.load_state_dict(torch.load(generator_pth, map_location=torch.device('cuda:'+str(gpu))))
    generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator).cuda()
    generator.cuda(gpu)
    generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[gpu])
    generator.eval()
    for param in generator.parameters():
        param.requires_grad = False
        
    autothresholder, *_ = model_selection(modelname=NAME_AUT, num_out_classes=2)
    autothresholder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(autothresholder).cuda()
    autothresholder.cuda(gpu)
    
    # load loss and optim
    ce = nn.CrossEntropyLoss().to(gpu)
    optim = get_optimizer('adam')(autothresholder.parameters(), lr = 0.0002, weight_decay= 0.00001)
    autothresholder, optim = amp.initialize(autothresholder, optim)
    autothresholder = torch.nn.parallel.DistributedDataParallel(autothresholder, device_ids=[gpu])
    scheduler = get_scheduler(optim, {'name': "StepLR", 'step_size': 20, 'gamma': 0.1})
    
    for epoch in range(num_epoch):
        # train
        autothresholder.train()
        ce.train()
        
        if gpu == 0:
            dataloader = tqdm(dataloader, position=0, leave=True, dynamic_ncols=True)
        
        atk = torchattacks.PGD(model, eps = 4/255, steps=5)
        atk.set_normalization_used(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            
        for i, _ in enumerate(dataloader):
            image, targets = _
                
            if torch.cuda.is_available():
                image = image.to(gpu)
                targets = targets.to(gpu)
            
            feat = model(image)[1]
            img = generator(feat, image)
            
            adv_imges = atk(image, targets)
            
            feat_adv = model(adv_imges)[1]
            adv_img = generator(feat_adv, image)
            
            samples = torch.cat((img, adv_img), dim = 0)
            labels = torch.cat((torch.zeros_like(targets), 1 - torch.zeros_like(targets)), dim = 0)

            preds = autothresholder(samples)[0]
                                
            optim.zero_grad()
            loss = ce(preds, labels)
            with amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
            optim.step()

            if gpu == 0:
                current_lr = optim.state_dict()['param_groups'][0]['lr']
                dataloader.set_description(
                    'Epoch: {Epoch:d}|lr: {lr:.7f}|loss: {loss:.8f}'.format(
                        Epoch = epoch, 
                        loss = loss.item(),
                        lr = current_lr))
            
        # validation
        if (epoch+1) % args.interval == 0:
            autothresholder.eval()
            acc.reset()
            acc_adv.reset()
            
            atk = torchattacks.PGD(model, eps = 4/255, steps=5)
            atk.set_normalization_used(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            
            for i, _ in enumerate(tqdm(dataloader_test)):
                image, targets = _
                
                if i % 2 != 0:
                    continue
                
                if torch.cuda.is_available():
                    image = image.to(gpu)
                    targets = targets.to(gpu)
                
                if i < (dataloader_test.__len__()//2):
                    with torch.no_grad():
                        pred = autothresholder(image)[0]
                        label = torch.zeros_like(targets)
                        acc.update(pred, label)

                else:
                    adv_imges = atk(image, targets)
                    
                    with torch.no_grad():
                        feat_adv = model(adv_imges)[1]
                        gen_adv_img = generator(feat_adv, image)
                        
                        pred_adv = autothresholder(gen_adv_img)[0]
                        label_adv = 1 - torch.zeros_like(targets)
                        
                        acc_adv.update(pred_adv, label_adv)
                    

            if gpu == 0:
                acc_epoch = acc.mean_acc()
                acc_adv_epoch = acc_adv.mean_acc()
                
                acc_avg = (acc_epoch + acc_adv_epoch) / 2
                
                if acc_avg > acc_save:
                    acc_save = acc_avg
                    torch.save(autothresholder.module.state_dict(), os.path.join(spth, NAME_AUT, NAME_DEC, 'best_'+ date +'.pth'))
                
                print('ACC:%.4f, ACC_ADV:%.4f, Best_ACC:%.4f' % (acc_epoch, acc_adv_epoch, acc_save))
                
                torch.save(autothresholder.module.state_dict(), os.path.join(spth, NAME_AUT, NAME_DEC, 'last_'+ date +'.pth'))
        
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