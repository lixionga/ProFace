import yaml
import argparse

import torch
import torch.nn as nn
import os
import yaml
import numpy as np
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
from trainer.utils import AccMeter
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
    dataset = FaceForensics(config)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank) #
    dataloader = data.DataLoader(dataset, batch_size=args.batchsize, shuffle=(train_sampler is None), num_workers=8, sampler=train_sampler)

    dataset_test = FaceForensics(config_test)
    dataloader_test = data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8)

    # train
    lpth = args.load_path
    spth = args.save_path
    num_epoch = args.epoch
    date = datetime.now().strftime('%Y%m%d%H%M')

    acc = AccMeter()
    acc_max = 0
    max_eps = 0.001 
    
    # load loss and optim and detector model
    NAME_DEC = args.detector
    NAME_DATA = args.dataset
    
    detector, *_ = model_selection(modelname = NAME_DEC, num_out_classes=2)
    detector_pth = os.path.join(lpth, NAME_DEC, NAME_DATA, 'best.pth')
    detector.load_state_dict(torch.load(detector_pth, map_location = torch.device('cuda:'+str(gpu))))
    detector = torch.nn.SyncBatchNorm.convert_sync_batchnorm(detector).cuda()
    detector.cuda(gpu)
    detector = torch.nn.parallel.DistributedDataParallel(detector, device_ids=[gpu])
    detector.eval()
    
    for param in detector.parameters():
        param.requires_grad = False
    
    NAME_COR = args.corrector
    corrector, *_ = model_selection(modelname=NAME_COR, num_out_classes=2)
    corrector = torch.nn.SyncBatchNorm.convert_sync_batchnorm(corrector).cuda()
    corrector.cuda(gpu)
    corrector = torch.nn.parallel.DistributedDataParallel(corrector, device_ids=[gpu])
    
    ce = nn.BCELoss().to(gpu)
    
    optim = get_optimizer('adam')(corrector.parameters(), lr = 0.0002, weight_decay= 0.00001)
    eps_list = [0, 0.0002, 0.0004, 0.0006, 0.001] 
    
    for epoch in range(num_epoch):
        # train
        corrector.train()
        ce.train()
        
        adv_fail = 0
        adv_success = 0
        
        if gpu == 0:
            dataloader = tqdm(dataloader, position=0, leave=True, dynamic_ncols=True)
            
        for i, _ in enumerate(dataloader):
            image, targets = _
            
            dynamic_eps = random.random() * max_eps
            
            atk = torchattacks.PGD(detector, eps = dynamic_eps, alpha=2/255, steps=5)
            atk.set_normalization_used(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            
            if torch.cuda.is_available():
                image = image.to(gpu)
                targets = targets.to(gpu)
            
            # generate adv success and fail sample
            preds = detector(image)[0]
            idx = torch.argmax(torch.softmax(preds, dim=-1), dim = -1).long() == targets
            
            image = image[idx==True]
            targets = targets[idx==True]

            adv_imges = atk(image, targets)
            
            adv_targets = torch.zeros_like(targets).float()
            adv_preds = detector(adv_imges)[0]
            
            adv_targets[torch.argmax(torch.softmax(adv_preds, dim=-1), dim = -1).long() != targets] = 1.0
            
            adv_preds_softmax = torch.softmax(adv_preds, dim=-1)
            transition_weight = torch.sigmoid(corrector(adv_imges)[0]).unsqueeze(1)
            adv_suc_pred = transition_weight @ adv_preds_softmax.unsqueeze(-1)
            adv_suc_pred = adv_suc_pred.squeeze(-1).squeeze(-1)

            optim.zero_grad()        
            loss = ce(adv_suc_pred, adv_targets)
            loss.backward()
            optim.step()

            if gpu == 0:
                current_lr = optim.state_dict()['param_groups'][0]['lr']
                dataloader.set_description(
                    'Epoch:{Epoch:d}|lr:{lr:.7f}|loss:{loss:.7f}|eps:{dynamic_eps:.6f}'.format(
                        Epoch = epoch, 
                        loss = loss.item(),
                        lr = current_lr,
                        dynamic_eps = dynamic_eps))
            
        # validation
        if (epoch+1) % args.interval == 0:
            corrector.eval()
            acc.reset()
            num_data = dataloader_test.__len__() // len(eps_list) 
            
            for i, _ in enumerate(tqdm(dataloader_test)):
                image, targets = _
                
                if i % 10 != 0:
                    continue
                
                if i / num_data < len(eps_list):
                    dynamic_eps = eps_list[i // num_data]
                else:
                    dynamic_eps = eps_list[len(eps_list) - 1]
                    
                atk = torchattacks.PGD(detector, eps = dynamic_eps, alpha = 2/255, steps = 5)

                atk.set_normalization_used(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
                    
                if torch.cuda.is_available():
                    image = image.to(gpu)
                    targets = targets.to(gpu)
                
                # generate adv success and fail sample
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
                
                if acc_epoch > acc_max:
                    acc_max = acc_epoch
                    torch.save(corrector.module.state_dict(), os.path.join(spth, NAME_COR, NAME_DEC, 'best_'+ date +'.pth'))
                
                print('fail num:%d, success num:%d' % (adv_fail, adv_success))
                print('eps:%.6f, ACC:%.4f, Best_ACC:%.4f' %(dynamic_eps, acc_epoch, acc_max))
                    

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    arg = arg_parser()
    config = arg.config
    
    if arg.dataset == 'FF++':
        config_path = "data_config/faceforensics.yml"
    elif arg.dataset == 'stargan':
        config_path = "data_config/stargan.yml"
    elif arg.dataset == 'stylegan':
        config_path = "data_config/stylegan.yml" 
        
    set_random_seed(arg.manual_seed, True)
    
    ngpus_per_node = torch.cuda.device_count()
    arg.ngpus_per_node = ngpus_per_node
    mp.spawn(main_worker, nprocs = ngpus_per_node, args = (arg, config, config_path))