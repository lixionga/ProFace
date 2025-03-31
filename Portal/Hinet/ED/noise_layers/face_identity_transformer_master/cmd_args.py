import sys
import os.path as osp
import numpy as np
import yaml
import argparse
from argparse import HelpFormatter
from operator import attrgetter

dataset_names = ['LFW', 'CASIA', 'LFW_CROP', 'FFHQ']


class SortingHelpFormatter(HelpFormatter):
    def add_arguments(self, actions):
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)


def define_parser():
    parser = argparse.ArgumentParser(description='Face Identity Transformer', formatter_class=SortingHelpFormatter)
    # compulsory
    # parser.add_argument('config_path', type=str,
    #                     default="/home/ysc/HiNet/face_identity_transformer_master/config/main.yaml")

    # for evaluation
    parser.add_argument('--inference_full', dest='evaluate', action='store_true',
                        help='inference model on full test set')
    parser.add_argument('--ckpt_name', type=str)

    # basics
    parser.add_argument('--passwd_length', type=int, default=16)
    parser.add_argument('--ckpt_dir', default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--not_resume_optimizer', action='store_false', dest='resume_optimizer')

    # data related
    parser.add_argument('--dataset', default='FFHQ', choices=dataset_names, help='dataset: ' + ' | '.join(dataset_names))
    # parser.add_argument('--data_root', default=osp.expanduser('~/data/face_datasets'), help='path to dataset')
    # parser.add_argument('--data_root', default=osp.expanduser('~/dataset'), help='path to dataset')
    parser.add_argument('--data_root', default=osp.expanduser('/Data/FFHQ256/image/test'), help='path to dataset')
    parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')

    # arch related
    parser.add_argument('--input_nc', type=int, default=3)
    parser.add_argument('--output_nc', type=int, default=3)
    parser.add_argument('--which_model_netD', type=str, default='multiscale_separated',
                        help='selects model to use for netD')
    parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks',
                        help='selects model to use for netG')
    parser.add_argument('--use_leakyG', action='store_true')
    parser.add_argument('--use_minus_one', type=str, default='half')
    parser.add_argument('--use_resize_conv', action='store_true')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--n_downsample_G', type=int, default=2, help='how many times 2x downsampling in generator')
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
    parser.add_argument('--n_layers_Q', type=int, default=3, help='')
    parser.add_argument('--normG', type=str, default='batch', help='instance normalization or batch normalization')
    parser.add_argument('--normD', type=str, default='batch', help='instance normalization or batch normalization')
    parser.add_argument('--normQ', type=str, default='batch', help='instance normalization or batch normalization')
    parser.add_argument('--padding_type', type=str, default='reflect')
    parser.add_argument('--gan_mode', type=str, default='lsgan',
                        help='the type of GAN objective. [vanilla|lsgan]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')

    parser.add_argument('--dropout', dest='dropout', action='store_true', help='add dropout for the generator')
    parser.add_argument('--pool_size', type=int, default=500, help='buffer size for the image buffer')

    # weight init
    parser.add_argument('--init_type', type=str, default='normal',
                        help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02,
                        help='scaling factor for normal, xavier and orthogonal.')

    # training related
    parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight decay. Junyan use 0')

    # loss weights
    parser.add_argument('--lambda_GAN', type=float, default=1.0)
    parser.add_argument('--lambda_GAN_recon', type=float, default=1.0)
    parser.add_argument('--lambda_GAN_M', type=float, default=1.0)
    parser.add_argument('--lambda_GAN_WR', type=float, default=1.0)

    parser.add_argument('--lambda_FR', type=float, default=1.0)
    parser.add_argument('--lambda_FR_M', type=float, default=1.0)
    parser.add_argument('--lambda_FR_WR', type=float, default=1.0)

    parser.add_argument('--lambda_dis', type=float, default=1.0)
    parser.add_argument('--lambda_G_recon', type=float, default=1.0)
    parser.add_argument('--lambda_L1', type=float, default=1.0)
    parser.add_argument('--lambda_rand_recon_L1', type=float, default=1.0)

    parser.add_argument('--lambda_Feat', type=float, default=1.0)
    parser.add_argument('--lambda_WR_Feat', type=float, default=0.0)
    parser.add_argument('--lambda_false_recon_diff', type=float, default=1.0,
                        help='weight for feature loss between fake and wrong recoon')

    # FR loss
    parser.add_argument('--feature_layer', type=int, default=5)
    parser.add_argument('--feature_loss', type=str, default='cos', choices=['cos', 'l2'])

    # for visualizer
    parser.add_argument('--auto_name', action='store_true')
    parser.add_argument('--name_add_on', type=str, default='')
    parser.add_argument('--name', type=str, default='experiment_name',
                        help='name of the experiment. It is displayed in Visdom and the HTML title')
    parser.add_argument('--no_html', action='store_true',
                        help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
    parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display, need change')
    parser.add_argument('--display_ncols', type=int, default=7,
                        help='if positive, display all images in a single visdom web panel with certain number of images per row.')
    parser.add_argument('--display_id', type=int, default=1, help='window id of the web display, usually never changed')
    parser.add_argument('--display_server', type=str, default="http://localhost",
                        help='visdom server of the web display, no need to change')

    # freqs
    parser.add_argument('--visdom_visual_freq', type=int, default=32,
                        help='frequency of showing qualitative training results on screen, to visdom')
    parser.add_argument('--plot_loss_freq', type=int, default=32,
                        help='frequency of showing training results on console and visdom')
    parser.add_argument('--print_loss_freq', type=int, default=10,
                        help='frequency of showing training results on console and visdom')
    parser.add_argument('--update_html_freq', type=int, default=64,
                        help='frequency of saving training results to html (CycleGAN impl)')
    parser.add_argument('--save_iter_freq', type=int, default=64,
                        help='iter frequency of saving the latest results during training')
    parser.add_argument('--html_iter_freq', type=int, default=1000,
                        help='iter freq to save my test html during training')
    parser.add_argument('--save_epoch_freq', type=int, default=1, help='epoch frequency of saving the latest results')
    parser.add_argument('--html_epoch_freq', type=int, default=1,
                        help='epoch freq to save my test html')
    parser.add_argument('--html_per_train_epoch', type=int, default=10,
                        help='how many visual results for html per training epoch,'
                             'for training mode only'
                             'for debug mode, no need to save multiple htmls per epoch')

    # test html
    parser.add_argument('--test_size', type=int, default=1,
                        help='# imgs to display per epoch for test, should fit in GPU mem, since we use batch norm, best to use the same training batch size')
    parser.add_argument('--testImageSize', type=int, default=128)
    parser.add_argument('--num_html_columns', type=int, default=3,
                        help='# columns of visual *set* to display per html')

    opt = parser.parse_args()
    return opt


def num2str(num):
    if num % 1 == 0.0:
        return str(int(num))
    else:
        return str(num).replace('.', '_')


def postprocess(args):
    args.start_epoch = 0

    if args.auto_name:
        args.name = '_'.join([args.ckpt_dir.strip('/').split('/')[-1],
                              num2str(args.lambda_Feat) + 'Feat',
                              num2str(args.lambda_WR_Feat) + 'WRFeat',
                              num2str(args.lambda_false_recon_diff) + 'MWFeat',
                              num2str(args.lambda_FR) + 'FR',
                              num2str(args.lambda_FR_M) + 'M',
                              num2str(args.lambda_FR_WR) + 'WR',
                              num2str(args.lambda_dis) + 'dis',
                              num2str(args.lambda_GAN) + 'GAN',
                              num2str(args.lambda_GAN_recon) + 'recon',
                              num2str(args.lambda_GAN_M) + 'M',
                              num2str(args.lambda_GAN_WR) + 'WR',
                              num2str(args.lambda_L1) + 'L1',
                              num2str(args.lambda_rand_recon_L1) + 'randreconL1',
                              num2str(args.lambda_G_recon) + 'recon'
                              ])
        if args.name_add_on != '':
            args.name += '_' + args.name_add_on
        if 'test' in sys.argv[0]:
            args.name = 'test_' + args.name
    args.ckpt_dir = osp.join(args.ckpt_dir, args.name)

    if not hasattr(args, 'during_training'):
        if 'test' in sys.argv[0]:
            args.during_training = False  # use images from test set during testing
        else:
            args.during_training = True  # sample images from val set during training for validation

    args.test_size = args.batch_size  # in order to not worry the effect of batch norm on diff batch_size
    return args


def parse_args(yaml_path):
    """
    parsing arguments: 1. cmd line 2. yaml 3. postprocess. The latter overwrites the former
    :param yaml_path
    :return: args
    """

    args = define_parser()

    with open(yaml_path, 'r') as fd:
        yaml_dict = yaml.load(fd, Loader=yaml.FullLoader)
        for k, v in yaml_dict.items():
            vars(args)[k] = v
    args = postprocess(args)
    return args
