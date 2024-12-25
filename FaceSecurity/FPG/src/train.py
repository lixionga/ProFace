import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import os
import random
from utils.fpg import Dataset
from utils.scheduler import LinearDecayLR
from sklearn.metrics import roc_auc_score
import argparse
from utils.logs import log
from utils.funcs import load_json, shape_refinemnet, magnitude_refinement
from datetime import datetime
from tqdm import tqdm
from model import Detector
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from apex import amp
from loss_pkgs import MLLoss_sim
from utils.qnet.eval import network, load_quality

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

def compute_accuray(pred,true):
    pred_idx=pred.argmax(dim=1).cpu().data.numpy()
    tmp=pred_idx==true.cpu().numpy()
    return sum(tmp)/len(pred_idx)

def main(gpu, args, config):
    cfg=load_json(config)

    args.rank = args.rank * args.ngpus_per_node + gpu
    
    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    torch.cuda.set_device(gpu)
    print(f"{args.dist_url}, ws:{args.world_size}, rank:{args.rank}")
    
    fake_local_label = False
    fake_target_label = False
    
    image_size = cfg['image_size'] 
    batch_size = cfg['batch_size']
    train_dataset = Dataset(phase='train', image_size = image_size, fake_target = fake_target_label, fake_local = fake_local_label)
    val_dataset = Dataset(phase='val', image_size = image_size)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank)
        
    train_loader=torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size//2,
                        shuffle=(train_sampler is None),
                        collate_fn=train_dataset.collate_fn,
                        num_workers=2,
                        pin_memory=False,
                        drop_last=True,
                        worker_init_fn=train_dataset.worker_init_fn,
                        sampler=train_sampler
                        )
    val_loader=torch.utils.data.DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=val_dataset.collate_fn,
                        num_workers=2,
                        pin_memory=False,
                        worker_init_fn=val_dataset.worker_init_fn
                        )
    
    model= Detector()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    model.cuda(gpu)
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, betas = (0.9, 0.98), eps = 1e-6, weight_decay = 0.2) 
    model, optimizer = amp.initialize(model, optimizer)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
        
    qnet = network('src/utils/qnet/model/pth', device = 'cuda:'+str(gpu))
    
    iter_loss=[]
    train_losses=[]
    sub_train_losses=[]
    sub_train_losses2=[]
    sub_train_losses3=[]
    train_accs=[]
    val_accs=[]
    val_losses=[]
    n_epoch=cfg['epoch']
    lr_scheduler=LinearDecayLR(optimizer, n_epoch, int(n_epoch/4*3))

    if gpu == 0 and args.weight is None:
        now=datetime.now()
        save_path='output/{}_'.format(args.session_name)+now.strftime(os.path.splitext(os.path.basename(args.config))[0])+'_'+now.strftime("%m_%d_%H_%M_%S")+'/'
        os.mkdir(save_path)
        os.mkdir(save_path+'weights/')
        os.mkdir(save_path+'logs/')
        logger = log(path=save_path+"logs/", file="losses.logs")

    criterion = nn.CrossEntropyLoss().to(gpu)

    target_layers = [model.module.net._blocks[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    targets = [ClassifierOutputTarget(1)] * (batch_size // 2)
    targets_real = [ClassifierOutputTarget(0)] * (batch_size // 2)

    fake_idx = torch.Tensor([i for i in range(batch_size) if i >= batch_size//2]).long().to(gpu)
    real_idx = torch.Tensor([i for i in range(batch_size) if i < batch_size//2]).long().to(gpu)
    idx_all = torch.Tensor([i for i in range(batch_size) if i < batch_size]).long().to(gpu)

    last_val_auc=0
    weight_dict={}
    n_weight=8
    mlloss = MLLoss_sim()
    mse = nn.MSELoss().train()
    regula_label = 0
    local = True
    cam_label = True
    local_storages = None
    
    for epoch in range(n_epoch):
        train_loss=0.
        sub_train_loss=0.
        sub_train_loss_2=0.
        sub_train_loss_3=0.
        train_acc=0.
        
        if args.weight == None:
            model.train(mode=True)
        else:
            model.eval()
        
        if gpu == 0:
            train_loader = tqdm(train_loader, position=0, leave=True, dynamic_ncols=True)
        
        for step, data in enumerate(train_loader):
            img = data['img'].to(gpu, non_blocking=True).float()
            target = data['label'].to(gpu, non_blocking=True).long()
            mask = data['mask'].to(gpu, non_blocking=True).float()
            source = data['source'].to(gpu, non_blocking=True).float()
            
            
            if fake_local_label:
                img_fake = data['local_fake'].to(gpu, non_blocking=True).float()
                local_mask = data['local_fake_mask'].to(gpu, non_blocking=True).float()
                local_storages = (img_fake, local_mask)

            if cam_label:
                model.eval()
                grayscale_cam_fake = cam(input_tensor=img[fake_idx], targets=targets)
                grayscale_cam_real = cam(input_tensor=img[real_idx], targets=targets_real)
                
                grayscale_cam = np.concatenate((grayscale_cam_real, grayscale_cam_fake), axis = 0)
                
                with torch.no_grad():
                    b = mask.shape[0]
                    mask[:b//2] = mask[:b//2] + mask[b//2:]
                    ref_outputs = model(img)
                    
                    img, mask = shape_refinemnet(img, grayscale_cam, mask, idx_all, model, ref_outputs, threshold = args.t, condition = local_storages)
                    
                img = magnitude_refinement(img, source, mask, target, model, EPS = args.eps, gpu = gpu)
                
                with torch.no_grad():
                    qualities = load_quality(qnet, img)
                
                model.train()

            if regula_label:
                output = model(img)
                loss2 = 0
                loss3 = 0
                
            elif local:
                output, feat = model(img, True)
                output, pred_q = output[0], output[1]

                if not cam_label:
                    b = mask.shape[0]
                    mask[:b//2] = mask[:b//2] + mask[b//2:]
                
                feat_0 = feat[0]
                
                mask[mask > 0] = 1 
                mask_resize = F.interpolate(mask, size=(feat_0.shape[2], feat_0.shape[3]))
                
                idx = torch.sum(mask_resize, dim=(1,2,3)) > 0
                mask_final = mask_resize[idx]

                feat_0_eff = feat_0[idx]
                c_final = feat_0_eff.shape[1]
                mask_final = mask_final.expand(-1, c_final, -1, -1)
                
                feat_0_in = torch.sum(feat_0_eff * mask_final, dim=(2, 3)) / torch.sum(mask_final, dim=(2, 3))
                feat_0_in = feat_0_in / feat_0_in.norm(dim=-1, keepdim=True)

                loss2 = mlloss(feat_0_in)
                loss3 = mse(pred_q[idx], qualities[idx])
                
            else:
                output = model(img)
                loss2 = 0
                loss3 = 0
            
            loss = criterion(output[idx], target[idx]) + args.alpha * loss2 + loss3
                
            if args.weight == None: 
                optimizer.zero_grad()
                with amp.scale_loss(loss, optimizer) as scaled_loss:                     
                    scaled_loss.backward()
                optimizer.step()
            
            loss_value = loss.item()
            sub_loss_value = loss2.item()
            sub_loss_value_2 = loss3.item()
            iter_loss.append(loss_value)
            
            train_loss += loss_value
            sub_train_loss += sub_loss_value
            sub_train_loss_2 += sub_loss_value_2
            
            acc=compute_accuray(F.log_softmax(output,dim=1),target)
            train_acc+=acc
        
            
        lr_scheduler.step()
        train_losses.append(train_loss/len(train_loader))
        sub_train_losses.append(sub_train_loss/len(train_loader))
        sub_train_losses2.append(sub_train_loss_2/len(train_loader))
        sub_train_losses3.append(sub_train_loss_3/len(train_loader))
        train_accs.append(train_acc/len(train_loader))

        log_text="Epoch {}/{} | train loss: {:.4f} | subtrain loss: {:.4f}, subtrain loss_2: {:.4f}, subtrain loss_3: {:.4f}, train acc: {:.4f}, ".format(
                        epoch + 1,
                        n_epoch,
                        train_loss/len(train_loader),
                        sub_train_loss/len(train_loader),
                        sub_train_loss_2/len(train_loader),
                        sub_train_loss_3/len(train_loader),
                        train_acc/len(train_loader),
                        )

        model.eval()
        val_loss=0.
        val_acc=0.
        output_dict=[]
        target_dict=[]
        
        if gpu == 0:
            with torch.no_grad():
                for step, data in enumerate(tqdm(val_loader)):
                    img = data['img'].to(gpu, non_blocking=True).float()
                    target = data['label'].to(gpu, non_blocking=True).long()

                    output=model(img)
                    loss=criterion(output,target)
                        
                    loss_value=loss.item()
                    iter_loss.append(loss_value)
                    val_loss+=loss_value
                    acc=compute_accuray(F.log_softmax(output,dim=1),target)
                    val_acc+=acc
                    output_dict+=output.softmax(1)[:,1].cpu().data.numpy().tolist()
                    target_dict+=target.cpu().data.numpy().tolist()
                    
                val_losses.append(val_loss/len(val_loader))
                val_accs.append(val_acc/len(val_loader))
                val_auc=roc_auc_score(target_dict,output_dict)
                log_text+="val loss: {:.4f}, val acc: {:.4f}, val auc: {:.4f}".format(
                                val_loss/len(val_loader),
                                val_acc/len(val_loader),
                                val_auc
                                )

                if len(weight_dict)<n_weight:
                    save_model_path=os.path.join(save_path+'weights/',"{}_{:.4f}_val.tar".format(epoch+1,val_auc))
                    weight_dict[save_model_path]=val_auc
                    torch.save({
                            "model":model.module.state_dict(),
                            "optimizer":optimizer.state_dict(),
                            "epoch":epoch
                        },save_model_path)
                    last_val_auc=min([weight_dict[k] for k in weight_dict])

                elif val_auc>=last_val_auc:
                    save_model_path=os.path.join(save_path+'weights/',"{}_{:.4f}_val.tar".format(epoch+1,val_auc))
                    for k in weight_dict:
                        if weight_dict[k]==last_val_auc:
                            del weight_dict[k]
                            os.remove(k)
                            weight_dict[save_model_path]=val_auc
                            break
                    torch.save({
                            "model":model.module.state_dict(),
                            "optimizer":optimizer.state_dict(),
                            "epoch":epoch
                        },save_model_path)
                    last_val_auc=min([weight_dict[k] for k in weight_dict])
                
                logger.info(log_text)

 
if __name__=='__main__':


    parser=argparse.ArgumentParser(description="config")
    parser.add_argument("--config",
                        type=str,
                        default="src/configs/fpg/base.json",
                        help="Specified the path of configuration file to be used.")
    parser.add_argument('-n',dest='session_name')
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
    parser.add_argument('--weight', default=None, type=str,
                        help='exploarion of trained model')
    parser.add_argument('--debug', default='True', type=str)
    parser.add_argument('--t', default=0.2, type=float)
    parser.add_argument('--eps', default=0.01, type=float)
    parser.add_argument('--alpha', default=0.075, type=float)
    parser.add_argument('--condition', default=0, type=int)
    parser.add_argument('--buffer', default=False, type=bool)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--buffer', default=False, type=bool)
    
    args=parser.parse_args()
    config = args.config
    
    set_random_seed(args.manual_seed, True)
    
    ngpus_per_node = torch.cuda.device_count()
    args.ngpus_per_node = ngpus_per_node
    mp.spawn(main, nprocs = ngpus_per_node, args=(args, config))