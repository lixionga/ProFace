# system libraries
import os, sys  # actually used
import os.path as osp
import time
import shutil
import numpy as np
from PIL import Image
import gc
from collections import OrderedDict

import torch  # actually used
import torchvision.transforms as transforms

# libraries for timeout
import signal
from contextlib import contextmanager

# libraries within this package
from cmd_args import parse_args
from utils.tools import *
from utils.image_pool import ImagePool
from utils.visualizer import Visualizer
from utils.util import generate_code, get_feat_loss, infoGAN_input, save_model, set_requires_grad, alignment
import datasets
import models
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

TORCH_VERSION = torch.__version__
# assert TORCH_VERSION.startswith('1.1') or TORCH_VERSION.startswith('1.2') or TORCH_VERSION.startswith('1.3'),\
#     'torch verision {} has not been tested'.format(TORCH_VERSION)

def grid_sample(input, grid):
    if TORCH_VERSION.startswith('1.1'):
        return torch.nn.functional.grid_sample(input, grid)  # (B, 3, h, w)
    else:
        return torch.nn.functional.grid_sample(input, grid, align_corners=True)  # (B, 3, h, w)


def main():
    global args
    args = parse_args(sys.argv[1])

    # -------------------- default arg settings for this model --------------------
    # num of display cols for visdom used during training time
    # real, fake1, fake2, recon1, recon2, wr1, wr2
    args.display_ncols = 6

    # define norm, either a single norm or define normalization method for each network separately
    if hasattr(args, 'norm'):
        args.normD = args.norm
        args.normQ = args.norm
        args.normG = args.norm

    args.gpu_ids = [0, 1]
    args.device = torch.device('cuda:0')

    args.timestamp = time.strftime('%m%d%H%M%S', time.localtime())  # add timestamp to ckpt_dir
    args.ckpt_dir += '_' + args.timestamp

    # ================================================================================
    # define args before logging args
    # -------------------- init ckpt_dir, logging --------------------
    os.makedirs(args.ckpt_dir, mode=0o777, exist_ok=True)

    # init visu
    visualizer = Visualizer(args)

    # log all the settings
    visualizer.logger.log('sys.argv:\n' + ' '.join(sys.argv))
    for arg in sorted(vars(args)):
        visualizer.logger.log('{:20s} {}'.format(arg, getattr(args, arg)))
    visualizer.logger.log('')

    # -------------------- code copy --------------------
    # copy config yaml
    shutil.copyfile(sys.argv[1], osp.join(args.ckpt_dir, osp.basename(sys.argv[1])))

    # TODO: delete after clean up!
    repo_basename = osp.basename(osp.dirname(osp.abspath(__file__)))
    repo_path = osp.join(args.ckpt_dir, repo_basename)
    os.makedirs(repo_path, mode=0o777, exist_ok=True)

    walk_res = os.walk('.')
    useful_paths = [path for path in walk_res if
                    '.git' not in path[0] and
                    'checkpoints' not in path[0] and
                    'configs' not in path[0] and
                    '__pycache__' not in path[0] and
                    'tee_dir' not in path[0] and
                    'tmp' not in path[0]]
    for p in useful_paths:
        for item in p[-1]:
            if not (item.endswith('.py') or item.endswith('.c') or item.endswith('.h') or item.endswith('.md')):
                continue
            old_path = osp.join(p[0], item)
            new_path = osp.join(repo_path, p[0][2:], item)
            basedir = osp.dirname(new_path)
            os.makedirs(basedir, mode=0o777, exist_ok=True)
            shutil.copyfile(old_path, new_path)

    # -------------------- dataset & loader --------------------
    train_dataset = datasets.__dict__[args.dataset](
        train=True,
        transform=transforms.Compose([
            transforms.Resize((args.imageSize, args.imageSize), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5), inplace=True)
        ]),
        args=args
    )
    visualizer.logger.log('train_dataset: ' + str(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

    # change test html / ckpt saving frequency
    args.html_iter_freq = len(train_loader) // args.html_per_train_epoch
    visualizer.logger.log('change args.html_iter_freq to %s' % args.html_iter_freq)
    args.save_iter_freq = len(train_loader) // args.html_per_train_epoch
    visualizer.logger.log('change args.save_iter_freq to %s' % args.html_iter_freq)

    val_dataset = datasets.__dict__[args.dataset](
        train=False,
        transform=transforms.Compose([
            transforms.Resize((args.imageSize, args.imageSize), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ]),
        args=args
    )
    visualizer.logger.log('val_dataset: ' + str(val_dataset))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

    # ================================================================================
    # -------------------- create model --------------------
    model_dict = dict()
    model_dict['D_nets'] = []
    model_dict['G_nets'] = []

    # D, Q
    assert args.which_model_netD == "multiscale_separated"
    D_names = ['M', 'R', 'WR']
    infogan_func = models.define_infoGAN
    model_dict['D'], model_dict['Q'] = infogan_func(
        args.output_nc,
        args.ndf,
        args.which_model_netD,
        args.n_layers_D,
        args.n_layers_Q,
        16,  # num_dis_classes: since we group 4 binary bits, 2^4 = 16.
        args.passwd_length // 4,
        args.normD,
        args.normQ,
        args.init_type,
        args.init_gain,
        D_names=D_names)

    model_dict['G_nets'].append(model_dict['Q'])
    model_dict['D_nets'].append(model_dict['D'])

    # G
    model_dict['G'] = models.define_G(args.input_nc+args.passwd_length, args.output_nc,
                                      args.ngf, args.which_model_netG, args.n_downsample_G,
                                      args.normG, args.dropout,
                                      args.init_type, args.init_gain,
                                      args.passwd_length,
                                      use_leaky=args.use_leakyG,
                                      use_resize_conv=args.use_resize_conv,
                                      padding_type=args.padding_type,
                                      )
    model_dict['G_nets'].append(model_dict['G'])

    # FR
    netFR = models.sphere20a(feature=args.feature_layer)
    netFR = torch.nn.DataParallel(netFR).cuda()
    netFR.module.load_state_dict(torch.load('/tmp/passwordGAN/pretrained_models/sphere20a_20171020.pth', map_location='cpu'))
    model_dict['FR'] = netFR
    model_dict['D_nets'].append(netFR)

    # log all the models
    visualizer.logger.log('model_dict')
    for k, v in model_dict.items():
        visualizer.logger.log(k + ':')
        if isinstance(v, list):
            visualizer.logger.log('list, len: ' + str(len(v)))
            for item in v:
                visualizer.logger.log(item.module.__class__.__name__, end=' ')
            visualizer.logger.log('')
        else:
            visualizer.logger.log(v)

    # -------------------- criterions --------------------
    criterion_dict = {
        'GAN': models.GANLoss(args.gan_mode).to(args.device),
        'FR': models.AngleLoss().to(args.device),
        'L1': torch.nn.L1Loss().to(args.device),
        'DIS': torch.nn.CrossEntropyLoss().to(args.device),
        'Feat': torch.nn.CosineEmbeddingLoss().to(args.device) if args.feature_loss == 'cos' else torch.nn.MSELoss().to(
            args.device)
    }

    # -------------------- optimizers --------------------
    # considering separate optimizer for each network?
    if not hasattr(args, 'G_lr'):
        args.G_lr = args.lr
    if not hasattr(args, 'Q_lr'):
        args.Q_lr = args.lr
    if not hasattr(args, 'D_lr'):
        args.D_lr = args.lr
    if not hasattr(args, 'FR_lr'):
        args.FR_lr = args.lr * 0.1

    optimizer_G_params = [{'params': model_dict['G'].parameters(), 'lr': args.G_lr},
                          {'params': model_dict['Q'].parameters(), 'lr': args.Q_lr}]
    optimizer_G = torch.optim.Adam(optimizer_G_params,
                                   betas=(args.beta1, 0.999),
                                   weight_decay=args.weight_decay)

    optimizer_D_params = [{'params': model_dict['D'].parameters(), 'lr': args.D_lr},
                          {'params': netFR.parameters(), 'lr': args.FR_lr}]
    optimizer_D = torch.optim.Adam(optimizer_D_params,
                                   betas=(args.beta1, 0.999),
                                   weight_decay=args.weight_decay)

    optimizer_dict = {
        'G': optimizer_G,
        'D': optimizer_D,
    }

    # -------------------- resume --------------------
    if args.resume:
        if osp.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            # better to restore to the exact iteration. I'm lazy here.
            args.start_epoch = checkpoint['epoch'] + 1

            for name, net in model_dict.items():
                if isinstance(net, list):
                    continue
                if hasattr(args, 'not_resume_models') and (name in args.not_resume_models):
                    continue
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                if 'state_dict_' + name in checkpoint:
                    try:
                        net.load_state_dict(checkpoint['state_dict_' + name])
                    except Exception as e:
                        visualizer.logger.log('fail to load model ' + name + ' ' + str(e))
                else:
                    visualizer.logger.log('model ' + name + ' not in checkpoints, just skip')

            if args.resume_optimizer:
                for name, optimizer in optimizer_dict.items():
                    if 'optimizer_' + name in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_' + name])
                    else:
                        visualizer.logger.log('optimizer ' + name + ' not in checkpoints, just skip')

            print("=> loaded checkpoints '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoints found at '{}'".format(args.resume))
        gc.collect()

    # -------------------- miscellaneous --------------------
    torch.backends.cudnn.enabled = True

    # generated image pool/buffer
    M_pool = ImagePool(args.pool_size)
    R_pool = ImagePool(args.pool_size)
    WR_pool = ImagePool(args.pool_size)

    # generate fixed passwords for test
    fixed_z, _, fixed_rand_z, _, \
    fixed_inv_z, _, fixed_rand_inv_z, _, \
    fixed_rand_inv_2nd_z, _ = generate_code(
        args.passwd_length, args.test_size, args.device,
        inv=True, use_minus_one=args.use_minus_one, gen_random_WR=True)
    print('fixed_z')
    print(fixed_z)
    fixed = {
        'z': fixed_z,
        'rand_z': fixed_rand_z,
        'inv_z': fixed_inv_z,
        'rand_inv_z': fixed_rand_inv_z,
        'rand_inv_2nd_z': fixed_rand_inv_2nd_z
    }
    gc.collect()

    for epoch in range(args.start_epoch, args.num_epochs):
        visualizer.logger.log('epoch ' + str(epoch))
        # turn on train mode
        model_dict['G'].train()
        model_dict['Q'].train()
        model_dict['D'].train()
        model_dict['FR'].train()

        # train
        epoch_start_time = time.time()
        train(train_loader, model_dict, criterion_dict, optimizer_dict, M_pool, R_pool, WR_pool, visualizer,
              epoch, args, val_loader, fixed)
        epoch_time = time.time() - epoch_start_time
        message = 'epoch %s total time %s\n' % (epoch, epoch_time)
        visualizer.logger.log(message)
        gc.collect()

        # save model
        if epoch % args.save_epoch_freq == 0:
            save_model(epoch, model_dict, optimizer_dict, args, iter=len(train_loader), save_sep=True)

        # test model, save to html for visualization
        if epoch % args.html_epoch_freq == 0:
            validate(val_loader, model_dict, visualizer, epoch, args, fixed, iter=len(train_loader))
        gc.collect()


def train(train_loader, model_dict, criterion_dict, optimizer_dict, fake_pool, recon_pool, WR_pool, visualizer,
          epoch, args, val_loader, fixed):
    iter_data_time = time.time()

    for i, (img, label, landmarks, img_path) in enumerate(train_loader):
        if img.size(0) != args.batch_size:
            continue

        img_cuda = img.cuda(non_blocking=True)

        if i % args.print_loss_freq == 0:
            iter_start_time = time.time()
            t_data = iter_start_time - iter_data_time

        visualizer.reset()

        # min_value = torch.min(img)  # -1
        # max_value = torch.max(img)  # 1
        #
        # print("最小像素值:", min_value.item())
        # print("最大像素值:", max_value.item())
        # a = 1/0
        # -------------------- forward & get aligned --------------------
        theta = alignment(landmarks)
        grid = torch.nn.functional.affine_grid(theta, torch.Size((args.batch_size, 3, 112, 96)))

        # -------------------- generate password --------------------
        z, dis_target, rand_z, rand_dis_target, \
        inv_z, inv_dis_target, rand_inv_z, rand_inv_dis_target, \
        rand_inv_2nd_z, rand_inv_2nd_dis_target = generate_code(args.passwd_length,
                                                                args.batch_size,
                                                                args.device,
                                                                inv=True,
                                                                use_minus_one=args.use_minus_one,
                                                                gen_random_WR=True)
        print(z)
        a= 1/0
        real_aligned = grid_sample(img_cuda, grid)  # (B, 3, h, w)
        real_aligned = real_aligned[:, [2, 1, 0], ...]

        fake = model_dict['G'](img, z.cpu())
        fake_aligned = grid_sample(fake, grid)
        fake_aligned = fake_aligned[:, [2, 1, 0], ...]

        recon = model_dict['G'](fake, inv_z)
        recon_aligned = grid_sample(recon, grid)
        recon_aligned = recon_aligned[:, [2, 1, 0], ...]

        rand_fake = model_dict['G'](img, rand_z.cpu())
        rand_fake_aligned = grid_sample(rand_fake, grid)
        rand_fake_aligned = rand_fake_aligned[:, [2, 1, 0, ], ...]

        rand_recon = model_dict['G'](fake, rand_inv_z)
        rand_recon_aligned = grid_sample(rand_recon, grid)
        rand_recon_aligned = rand_recon_aligned[:, [2, 1, 0], ...]

        rand_recon_2nd = model_dict['G'](fake, rand_inv_2nd_z)
        rand_recon_2nd_aligned = grid_sample(rand_recon_2nd, grid)
        rand_recon_2nd_aligned = rand_recon_2nd_aligned[:, [2, 1, 0], ...]

        # init loss dict for plot & print
        current_losses = {}

        # -------------------- D PART --------------------
        # init
        set_requires_grad(model_dict['G_nets'], False)
        set_requires_grad(model_dict['D_nets'], True)
        optimizer_dict['D'].zero_grad()
        loss_D = 0.

        # ========== Face Recognition (FR) losses (L_{adv}, L_{rec\_cls}) ==========
        # FAKE FRs
        # M
        id_fake = model_dict['FR'](fake_aligned.detach())[0]
        loss_D_FR_fake = criterion_dict['FR'](id_fake, label.to(args.device))

        # R & WR
        id_recon = model_dict['FR'](recon_aligned.detach())[0]
        loss_D_FR_recon = -criterion_dict['FR'](id_recon, label.to(args.device))

        id_rand_recon = model_dict['FR'](rand_recon_aligned.detach())[0]
        loss_D_FR_rand_recon = criterion_dict['FR'](id_rand_recon, label.to(args.device))

        loss_D_FR_fake_total = args.lambda_FR_M * loss_D_FR_fake + loss_D_FR_recon \
                               + args.lambda_FR_WR * loss_D_FR_rand_recon
        loss_D_FR_fake_avg = loss_D_FR_fake_total / float(1. + args.lambda_FR_M + args.lambda_FR_WR)
        current_losses.update({
            'D_FR_M': loss_D_FR_fake.item(),
            'D_FR_R': loss_D_FR_recon.item(),
            'D_FR_WR': loss_D_FR_rand_recon.item(),
        })

        # REAL FR
        id_real = model_dict['FR'](real_aligned)[0]
        loss_D_FR_real = criterion_dict['FR'](id_real, label.to(args.device))

        loss_D += args.lambda_FR * (loss_D_FR_real + loss_D_FR_fake_avg) * 0.5
        current_losses.update({'D_FR_real': loss_D_FR_real.item(),
                               'D_FR_fake': loss_D_FR_fake_avg.item()})

        # ========== GAN loss (L_{GAN}) ==========
        # fake
        all_M = torch.cat((fake.detach().cpu(),
                           rand_fake.detach().cpu(),
                           ), 0)
        pred_D_M = model_dict['D'](fake_pool.query(all_M, batch_size=args.batch_size), 'M')
        loss_D_M = criterion_dict['GAN'](pred_D_M, False)

        # R
        pred_D_recon = model_dict['D'](recon_pool.query(recon.detach().cpu(), batch_size=args.batch_size), 'R')
        loss_D_recon = criterion_dict['GAN'](pred_D_recon, False)

        # WR
        all_WR = torch.cat((rand_recon.detach().cpu(),
                            rand_recon_2nd.detach().cpu()
                            ), 0)
        pred_D_WR = model_dict['D'](WR_pool.query(all_WR, batch_size=args.batch_size), 'WR')
        loss_D_WR = criterion_dict['GAN'](pred_D_WR, False)

        loss_D_fake_total = args.lambda_GAN_M * loss_D_M + \
                            args.lambda_GAN_recon * loss_D_recon + \
                            args.lambda_GAN_WR * loss_D_WR
        loss_D_fake_total_weights = args.lambda_GAN_M + \
                                    args.lambda_GAN_recon + \
                                    args.lambda_GAN_WR
        loss_D_GAN_fake = loss_D_fake_total / loss_D_fake_total_weights
        current_losses.update({
            'D_GAN_M': loss_D_M.item(),
            'D_GAN_R': loss_D_recon.item(),
            'D_GAN_WR': loss_D_WR.item()
        })

        # real
        pred_D_real_M = model_dict['D'](img, 'M')
        pred_D_real_R = model_dict['D'](img, 'R')
        pred_D_real_WR = model_dict['D'](img, 'WR')

        loss_D_real_M = criterion_dict['GAN'](pred_D_real_M, True)
        loss_D_real_R = criterion_dict['GAN'](pred_D_real_R, True)
        loss_D_real_WR = criterion_dict['GAN'](pred_D_real_WR, True)

        loss_D_GAN_real = (args.lambda_GAN_M * loss_D_real_M +
                           args.lambda_GAN_recon * loss_D_real_R +
                           args.lambda_GAN_WR * loss_D_real_WR) / \
                          (args.lambda_GAN_M +
                           args.lambda_GAN_recon +
                           args.lambda_GAN_WR)

        loss_D += args.lambda_GAN * (loss_D_GAN_fake + loss_D_GAN_real) * 0.5
        current_losses.update({
            'D_GAN_real': loss_D_GAN_real.item(),
            'D_GAN_fake': loss_D_GAN_fake.item()
        })
        current_losses['D'] = loss_D.item()

        # D backward and optimizer steps
        loss_D.backward()
        optimizer_dict['D'].step()


        # -------------------- G PART --------------------
        # init
        set_requires_grad(model_dict['D_nets'], False)
        set_requires_grad(model_dict['G_nets'], True)
        optimizer_dict['G'].zero_grad()
        loss_G = 0

        # ========== GAN loss (L_{GAN}) ==========
        pred_G_fake = model_dict['D'](fake, 'M')
        loss_G_GAN_fake = criterion_dict['GAN'](pred_G_fake, True)

        pred_G_recon = model_dict['D'](recon, 'R')
        loss_G_GAN_recon = criterion_dict['GAN'](pred_G_recon, True)

        pred_G_WR = model_dict['D'](rand_recon, 'WR')
        loss_G_GAN_WR = criterion_dict['GAN'](pred_G_WR, True)

        loss_G_GAN_total = args.lambda_GAN_M * loss_G_GAN_fake + \
                           args.lambda_GAN_recon * loss_G_GAN_recon + \
                           args.lambda_GAN_WR * loss_G_GAN_WR
        loss_G_GAN_total_weights = args.lambda_GAN_M + args.lambda_GAN_recon + args.lambda_GAN_WR
        loss_G_GAN = loss_G_GAN_total / loss_G_GAN_total_weights
        loss_G += args.lambda_GAN * loss_G_GAN

        current_losses.update({
            'G_GAN_M': loss_G_GAN_fake.item(),
            'G_GAN_R': loss_G_GAN_recon.item(),
            'G_GAN_WR': loss_G_GAN_WR.item(),
            'G_GAN': loss_G_GAN.item()})

        # ========== infoGAN loss (L_{aux}) ==========
        if args.lambda_dis > 0:
            fake_dis_logits = model_dict['Q'](infoGAN_input(img_cuda, fake))
            infogan_fake_acc = 0
            loss_G_fake_dis = 0
            for dis_idx in range(args.passwd_length // 4):
                a = fake_dis_logits[dis_idx].max(dim=1)[1]
                b = dis_target[:, dis_idx]
                acc = torch.eq(a, b).type(torch.float).mean()
                infogan_fake_acc += acc.item()
                loss_G_fake_dis += criterion_dict['DIS'](fake_dis_logits[dis_idx], dis_target[:, dis_idx])
            infogan_fake_acc = infogan_fake_acc / float(args.passwd_length // 4)

            recon_dis_logits = model_dict['Q'](infoGAN_input(fake, recon))
            infogan_recon_acc = 0
            loss_G_recon_dis = 0
            for dis_idx in range(args.passwd_length // 4):
                a = recon_dis_logits[dis_idx].max(dim=1)[1]
                b = inv_dis_target[:, dis_idx]
                acc = torch.eq(a, b).type(torch.float).mean()
                infogan_recon_acc += acc.item()
                loss_G_recon_dis += criterion_dict['DIS'](recon_dis_logits[dis_idx], inv_dis_target[:, dis_idx])
            infogan_recon_acc = infogan_recon_acc / float(args.passwd_length // 4)

            rand_recon_dis_logits = model_dict['Q'](infoGAN_input(fake, rand_recon))
            infogan_rand_recon_acc = 0
            loss_G_recon_rand_dis = 0
            for dis_idx in range(args.passwd_length // 4):
                a = rand_recon_dis_logits[dis_idx].max(dim=1)[1]
                b = rand_inv_dis_target[:, dis_idx]
                acc = torch.eq(a, b).type(torch.float).mean()
                infogan_rand_recon_acc += acc.item()
                loss_G_recon_rand_dis += criterion_dict['DIS'](rand_recon_dis_logits[dis_idx],
                                                               rand_inv_dis_target[:, dis_idx])
            infogan_rand_recon_acc = infogan_rand_recon_acc / float(args.passwd_length // 4)

            dis_loss_total = loss_G_fake_dis + loss_G_recon_dis + loss_G_recon_rand_dis
            dis_acc_total = infogan_fake_acc + infogan_recon_acc + infogan_rand_recon_acc
            dis_cnt = 3

            loss_G += args.lambda_dis * dis_loss_total
            current_losses.update({
                'dis': dis_loss_total.item(),
                'dis_acc': dis_acc_total / float(dis_cnt)})

        # ========== Face Recognition (FR) loss (L_{adv}, L{rec_cls}})==========
        # (netFR must not be fixed)
        id_fake_G, fake_feat = model_dict['FR'](fake_aligned)
        loss_G_FR_fake = -criterion_dict['FR'](id_fake_G, label.to(args.device))

        id_recon_G, recon_feat = model_dict['FR'](recon_aligned)
        loss_G_FR_recon = criterion_dict['FR'](id_recon_G, label.to(args.device))

        id_rand_recon_G, rand_recon_feat = model_dict['FR'](rand_recon_aligned)
        loss_G_FR_rand_recon = -criterion_dict['FR'](id_rand_recon_G, label.to(args.device))

        loss_G_FR_avg = (args.lambda_FR_M * loss_G_FR_fake +
                         loss_G_FR_recon +
                         args.lambda_FR_WR * loss_G_FR_rand_recon) /\
                        (args.lambda_FR_M + 1. + args.lambda_FR_WR)
        loss_G += args.lambda_FR * loss_G_FR_avg

        current_losses.update({
            'G_FR_M': loss_G_FR_fake.item(),
            'G_FR_R': loss_G_FR_recon.item(),
            'G_FR_WR': loss_G_FR_rand_recon.item(),
            'G_FR': loss_G_FR_avg.item()
        })

        # ========== Feature losses (L_{feat} is the sum of the three L_{dis}'s) ==========
        if args.feature_loss == 'cos':  # make cos sim target
            FR_cos_sim_target = torch.empty(size=(args.batch_size, 1), dtype=torch.float32, device=args.device)
            FR_cos_sim_target.fill_(-1.)
        else:
            FR_cos_sim_target = None

        FR_cos_sim_target = FR_cos_sim_target.squeeze()  # 新加的 否则cosembedding相似度会出错
        id_rand_fake_G, rand_fake_feat = model_dict['FR'](rand_fake_aligned)
        id_rand_recon_2nd_G, rand_recon_2nd_feat = model_dict['FR'](rand_recon_2nd_aligned)

        if args.lambda_Feat:
            loss_G_feat = get_feat_loss(fake_feat, rand_fake_feat, FR_cos_sim_target, args.feature_loss, criterion_dict)
            current_losses['G_feat'] = loss_G_feat.item()
        else:
            loss_G_feat = 0.

        if args.lambda_WR_Feat:
            loss_G_WR_feat = get_feat_loss(rand_recon_feat, rand_recon_2nd_feat, FR_cos_sim_target, args.feature_loss,
                                           criterion_dict)
            current_losses['G_WR_feat'] = loss_G_WR_feat.item()
        else:
            loss_G_WR_feat = 0.

        if args.lambda_false_recon_diff:
            loss_G_M_WR_feat = get_feat_loss(fake_feat, rand_recon_feat, FR_cos_sim_target, args.feature_loss,
                                             criterion_dict)
            current_losses['G_feat_M_WR'] = loss_G_M_WR_feat.item()
        else:
            loss_G_M_WR_feat = 0.

        loss_G += args.lambda_Feat * loss_G_feat + \
                  args.lambda_WR_Feat * loss_G_WR_feat + \
                  args.lambda_false_recon_diff * loss_G_M_WR_feat

        # ========== L1/Recon losses (L_1, L_{rec}) ==========
        loss_G_L1 = criterion_dict['L1'](fake, img_cuda)
        loss_G_rand_recon_L1 = criterion_dict['L1'](rand_recon, img_cuda)
        loss_G_recon = criterion_dict['L1'](recon, img_cuda)

        loss_G += args.lambda_L1 * loss_G_L1 + \
                  args.lambda_rand_recon_L1 * loss_G_rand_recon_L1 + \
                  args.lambda_G_recon * loss_G_recon

        current_losses.update({
            'L1_M': loss_G_L1.item(),
            'recon': loss_G_recon.item(),
            'L1_WR': loss_G_rand_recon_L1.item()
        })


        current_losses['G'] = loss_G.item()

        # G backward and optimizer steps
        loss_G.backward()
        optimizer_dict['G'].step()

        # -------------------- LOGGING PART --------------------
        if i % args.print_loss_freq == 0:
            t = (time.time() - iter_start_time) / args.batch_size
            visualizer.print_current_losses(epoch, i, current_losses, t, t_data)
            if args.display_id > 0 and i % args.plot_loss_freq == 0:
                visualizer.plot_current_losses(epoch, float(i) / len(train_loader), args, current_losses)

        if i % args.visdom_visual_freq == 0:
            save_result = i % args.update_html_freq == 0

            current_visuals = OrderedDict()
            current_visuals['real'] = img.detach()
            current_visuals['fake'] = fake.detach()
            current_visuals['rand_fake'] = rand_fake.detach()
            current_visuals['recon'] = recon.detach()
            current_visuals['rand_recon'] = rand_recon.detach()
            current_visuals['rand_recon_2nd'] = rand_recon_2nd.detach()

            try:
                with time_limit(60):
                    visualizer.display_current_results(current_visuals, epoch, save_result, args)
            except TimeoutException:
                visualizer.logger.log(
                    'TIME OUT visualizer.display_current_results epoch:{} iter:{}. Change display_id to -1'.format(
                        epoch, i))
                # disable visdom display ever since
                args.display_id = -1

        # +1 so that we do not save/test for 0th iteration
        if (i + 1) % args.save_iter_freq == 0:
            save_model(epoch, model_dict, optimizer_dict, args, iter=i, save_sep=False)
            if args.display_id > 0:
                visualizer.vis.save([args.name])

        if (i + 1) % args.html_iter_freq == 0:
            validate(val_loader, model_dict, visualizer, epoch, args, fixed, i)

        if (i + 1) % args.print_loss_freq == 0:
            iter_data_time = time.time()


def validate(val_loader, model_dict, visualizer, epoch, args, fixed, iter=0):
    with torch.no_grad():
        for i, (img, label, landmarks, img_path) in enumerate(val_loader):
            if args.use_minus_one:
                another_inv_z = fixed['rand_z'] * -1
            else:
                another_inv_z = 1.0 - fixed['rand_z']

            fake = model_dict['G'](img, fixed['z'].cpu())
            rand_fake = model_dict['G'](img, fixed['rand_z'])
            recon = model_dict['G'](fake, fixed['inv_z'])
            another_recon = model_dict['G'](rand_fake, another_inv_z)
            rand_recon = model_dict['G'](fake, fixed['rand_inv_z'])
            rand_recon_2nd = model_dict['G'](fake, fixed['rand_inv_2nd_z'])

            current_visuals = OrderedDict()
            current_visuals['real'] = img
            current_visuals['fake'] = fake
            current_visuals['rand_fake'] = rand_fake
            current_visuals['recon'] = recon
            current_visuals['another_recon'] = another_recon
            current_visuals['rand_recon'] = rand_recon
            current_visuals['rand_recon_2nd'] = rand_recon_2nd

            visualizer.display_test_results_html(current_visuals, img_path, epoch, iter, use_real=False)


if __name__ == '__main__':
    main()
