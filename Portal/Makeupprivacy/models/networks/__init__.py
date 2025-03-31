import torch
from Makeupprivacy.models.networks.losses import *
from Makeupprivacy.models.networks.discriminator import *
from Makeupprivacy.models.networks.generator import *
from Makeupprivacy.models.networks.correspondence import *
import Makeupprivacy.util.util as util


def find_network_using_name(target_network_name, filename):
    target_class_name = target_network_name + filename 
    module_name = 'Makeupprivacy.models.networks.' + filename 
    network = util.find_class_in_module(target_class_name, module_name)  

    return network


def modify_commandline_options(parser, is_train):
    opt, _ = parser.parse_known_args()

    netC_cls = find_network_using_name(opt.netC, 'correspondence')
    parser = netC_cls.modify_commandline_options(parser, is_train)
    netG_cls = find_network_using_name(opt.netG, 'generator')
    parser = netG_cls.modify_commandline_options(parser, is_train)
    if is_train:
        netD_cls = find_network_using_name(opt.netD, 'discriminator')
        parser = netD_cls.modify_commandline_options(parser, is_train)
    
    return parser


def create_network(cls, opt):
    net = cls(opt)
    net.print_network()
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
    net.init_weights(opt.init_type, opt.init_variance)
    return net


def define_C(opt):
    netC_cls = find_network_using_name(opt.netC, 'correspondence')
    netC = netC_cls(opt)
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netC.cuda()
    return netC


def define_G(opt):
    netG_cls = find_network_using_name(opt.netG, 'generator')
    return create_network(netG_cls, opt)


def define_D(opt):
    netD_cls = find_network_using_name(opt.netD, 'discriminator')
    return create_network(netD_cls, opt)