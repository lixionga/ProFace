import torch
import torch.nn as nn
from torch.nn import init
import functools
# import numpy as np
import gc


###############################################################################
# Helper Functions
###############################################################################
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=False)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02):
    net = torch.nn.DataParallel(net).cuda()

    gc.collect()
    torch.cuda.empty_cache()
    init_weights(net, init_type, gain=init_gain)
    return net


###############################################################################
# Network Define Functions
###############################################################################
def define_G(input_nc, output_nc, ngf, which_model_netG, n_downsample_G, norm='batch', use_dropout=False,
             init_type='normal', init_gain=0.02, passwd_length=16, use_leaky=False, use_resize_conv=False,
             padding_type='reflect'):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, n_downsampling=n_downsample_G, norm_layer=norm_layer,
                               use_dropout=use_dropout, n_blocks=9, use_leaky=use_leaky,
                               use_resize_conv=use_resize_conv, padding_type=padding_type,
                               )
    # elif which_model_netG == 'resnet_6blocks':
    #     netG = ResnetGenerator(input_nc, output_nc, ngf, n_downsampling=n_downsample_G, norm_layer=norm_layer,
    #                            use_dropout=use_dropout, n_blocks=6, use_leaky=use_leaky,
    #                            use_resize_conv=use_resize_conv)
    # elif which_model_netG == 'unet_128':
    #     netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    # elif which_model_netG == 'unet_256':
    #     netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, init_gain)


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02,
             D_names=['M', 'WR', 'R']):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    # if which_model_netD == 'basic':
    #     netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    # elif which_model_netD == 'n_layers':
    #     netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    # elif which_model_netD == 'pixel':
    #     netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    # elif which_model_netD == 'multiscale':
    #     netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    if which_model_netD == 'multiscale_separated':
        netD = MultiscaleDiscriminatorSharedFE(D_names, input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, init_gain)


def define_infoGAN(input_nc, ndf, which_model_netD, n_layers_D, n_layers_Q, num_dis_classes, dis_length, normD='batch',
                   normQ='batch', init_type='normal', init_gain=0.02,
                   D_names=['M', 'WR', 'R'],
                   ):
    """
    define net D, Q for infoGAN_based arch
    """
    netD = define_D(input_nc, ndf, which_model_netD,
                    n_layers_D=n_layers_D, norm=normD,
                    init_type=init_type, init_gain=init_gain, D_names=D_names,
                    )

    norm_layer_Q = get_norm_layer(norm_type=normQ)
    input_nc_Q = input_nc * 2
    netQ = MultiClassifier(input_nc_Q, ndf, n_layers_Q, norm_layer_Q, num_dis_classes, dis_length)

    return netD, init_net(netQ, init_type, init_gain)


##############################################################################
# Classes
##############################################################################

##############################################################################
# Losses
##############################################################################
# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
                Parameters:
                    gan_mode (string)-- the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
                    target_real_label (bool) -- label for a real image
                    target_fake_label (bool)-- label of a fake image
                Note: Do not use sigmoid as the last layer of Discriminator.
                LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def get_single_loss(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

    def __call__(self, input, target_is_real):
        if isinstance(input, list):
            loss = 0
            for input_i in input:
                loss += self.get_single_loss(input_i, target_is_real)
            return loss
        else:
            return self.get_single_loss(input, target_is_real)


##############################################################################
# Generator
##############################################################################

class MappingNet(nn.Module):
    def __init__(self, passwd_length, output_ncs):
        super(MappingNet, self).__init__()

        model = []

        input_nc = passwd_length
        for output_nc in output_ncs:
            model.append(nn.Linear(input_nc, output_nc))
            model.append(nn.LeakyReLU(negative_slope=0.2))
            input_nc = output_nc
        self.model = nn.Sequential(*model)

    def forward(self, password):
        """

        :param password: (B, passwd_length)
        :return: (B, output_ncs[-1])
        """
        return self.model(password)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=2, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=9, use_leaky=False, use_resize_conv=False, padding_type='reflect',
                 ):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        activation = nn.LeakyReLU(0.1, inplace=True) if use_leaky else nn.ReLU(True)

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf) if norm_layer is not None else nn.Sequential(),
                 activation]  # 128, ngf=64

        # n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.ReflectionPad2d(1),
                      nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=0, bias=use_bias),
                      norm_layer(ngf * mult * 2) if norm_layer is not None else nn.Sequential(),
                      activation]
            # when n_downsampling=2, 64 -> 128, 64; 128 -> 256, 32
            # when n_downsampling=2, 64 -> 128, 64

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias, use_leaky=use_leaky)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            if use_resize_conv:
                model += [nn.Upsample(scale_factor=2),
                          nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1, padding=0),
                          norm_layer(int(ngf * mult / 2)) if norm_layer is not None else nn.Sequential(),
                          activation]
            else:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)) if norm_layer is not None else nn.Sequential(),
                          activation]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]  # (-1, 1)

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        if z is not None:
            input = torch.cat((input, z.view(z.size(0), -1, 1, 1).repeat(1, 1, input.size(2), input.size(3))), 1)
        return self.model(input)
        # + input


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, use_leaky):
        super(ResnetBlock, self).__init__()
        self.activation = nn.LeakyReLU(0.1, inplace=True) if use_leaky else nn.ReLU(True)
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim) if norm_layer is not None else nn.Sequential(),
                       self.activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim) if norm_layer is not None else nn.Sequential()]
        # no ReLU and no dropout

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)  # diff: #input channels, use_dropout
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input, z=None):
        if z is not None:
            input = torch.cat((input, z.view(z.size(0), -1, 1, 1).repeat(1, 1, input.size(2), input.size(3))), 1)
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer == nn.BatchNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc) if norm_layer is not None else nn.Sequential()
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc) if norm_layer is not None else nn.Sequential()
        # down Leaky up normal ReLU???

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


##############################################################################
# Discriminator
##############################################################################


class MultiClassifier(nn.Module):
    def __init__(self, input_nc, ndf, n_layers, norm_layer, num_dis_classes, dis_length):
        super(MultiClassifier, self).__init__()
        self.FE = NLayerFrontEnd(input_nc, ndf, n_layers, norm_layer)
        self.classifiers = MultiClassifierAfterFE(ndf, n_layers, norm_layer, num_dis_classes, dis_length)

    def forward(self, x):
        intermediate = self.FE(x)
        return self.classifiers(intermediate)


class MultiClassifierAfterFE(nn.Module):
    def __init__(self, ndf, n_layers, norm_layer, num_dis_classes, dis_length):
        super(MultiClassifierAfterFE, self).__init__()
        self.dis_length = dis_length
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        kw = 4
        padw = 1

        nf_mult_prev = 2 ** (n_layers - 1)
        nf_mult = min(2 ** n_layers, 8)

        sequence = [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult) if norm_layer is not None else nn.Sequential(),
            nn.LeakyReLU(0.2, True),
            #  4 -- 8

            nn.Conv2d(ndf * nf_mult, ndf * nf_mult,
                      kernel_size=kw, stride=2 * 2, padding=0, bias=use_bias),
            #  5 -- 2
            norm_layer(ndf * nf_mult) if norm_layer is not None else nn.Sequential(),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2)]
        self.model = nn.Sequential(*sequence)

        self.fc_list = nn.ModuleDict()
        for dis_idx in range(dis_length):
            self.fc_list['fc_' + str(dis_idx)] = nn.Linear(ndf * nf_mult, num_dis_classes)

    def forward(self, input):
        intermediate = self.model(input)
        intermediate = intermediate.view(input.size(0), -1)
        results = []
        for dis_idx in range(self.dis_length):
            results.append(self.fc_list['fc_' + str(dis_idx)](intermediate))
        return results


class NLayerFrontEnd(nn.Module):
    def __init__(self, input_nc, ndf, n_layers, norm_layer):
        super(NLayerFrontEnd, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]  # 0 -- 64

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult) if norm_layer is not None else nn.Sequential(),
                nn.LeakyReLU(0.2, True)
            ]
        #  2 -- 32, 3 -- 16
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class MultiscaleDiscriminatorSharedFE(nn.Module):
    """
    return a list of patchGAN results for different Ds
    """

    def __init__(self, D_names, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(MultiscaleDiscriminatorSharedFE, self).__init__()
        self.n_layers = n_layers

        netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer)
        self.FE = netD.model

        for D_name in D_names:
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer)
            setattr(self, 'layer1_' + D_name, netD.model)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        return model(input)

    def forward(self, input, D_name):
        result = []
        input_downsampled = input

        model = getattr(self, 'layer1_' + D_name)
        result.append(self.singleD_forward(model, input_downsampled))

        input_downsampled = self.downsample(input_downsampled)
        result.append(self.singleD_forward(self.FE, input_downsampled))
        return result


class MultiscaleDiscriminator(nn.Module):
    """
    return a list of patchGAN results for different Ds
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 num_D=2):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer)
            setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        return model(input)

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        self.n_layers = n_layers
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        # padw = int(np.ceil((kw - 1.0) / 2))  # 2
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]  # 0 -- 65 or floor(x / 2 + 1) ; 33

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [
                nn.Conv2d(nf_prev, nf,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(nf) if norm_layer is not None else nn.Sequential(),
                nn.LeakyReLU(0.2, True)
            ]
        # 1 -- 33, 2 -- 17; 17, 9

        nf_prev = nf
        nf = min(nf * 2, 512)
        kw = 4  # if n_layers == 3 else 3
        sequence += [
            nn.Conv2d(nf_prev, nf,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(nf) if norm_layer is not None else nn.Sequential(),
            nn.LeakyReLU(0.2, True)
        ]
        #  3 -- 18; 10
        # only diff: stride

        kw = 4  # if n_layers == 3 else 3
        sequence += [nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw, bias=False)]
        #  4 -- 19; 11
        # rm bias=False

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2) if norm_layer is not None else nn.Sequential(),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)
